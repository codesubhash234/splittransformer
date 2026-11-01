import React, { useEffect, useRef, useState } from 'react';
import {
  MODEL_SPEC,
  NodeARuntime,
  NodeBRuntime,
  ShardAWeights,
  ShardBWeights,
  createShardAWeights,
  createShardBWeights,
  sampleNextByte,
  formatByteGlyph,
} from './engine/transformer';
import {
  DataChannelPacketAssembler,
  PacketType,
  encodeActivationFullChunks,
  encodeActivationStepPacket,
  encodeResetSessionPacket,
  encodeTokenResultPacket,
} from './engine/protocol';
import {
  ConnectionState,
  DataChannelStats,
  PeerRole,
  WebRTCMeshManager,
} from './engine/webrtc';
import { TorchMT19937 } from './engine/prng';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { PairingPanel } from './components/PairingPanel';
import { ArchitectureSpecPanel } from './components/ArchitectureSpecPanel';
import { DeterminismSuitePanel } from './components/DeterminismSuitePanel';

type ActiveTab = 'workspace' | 'pairing' | 'architecture' | 'determinism';

interface GeneratedByteRecord {
  step: number;
  seqPos: number;
  byte: number;
  nodeAMs: number;
  nodeBMs: number;
  rttMs: number;
  txBytes: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workspace');
  const [role, setRole] = useState<PeerRole>('dual-loopback');
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [channelStats, setChannelStats] = useState<DataChannelStats>({
    messagesSent: 0,
    messagesReceived: 0,
    bytesSent: 0,
    bytesReceived: 0,
    lastPacketBytes: 0,
    lastRttMs: 0,
    transportMode: 'p2p-webrtc-loopback',
    iceCandidateType: 'host (WebRTC SCTP loopback)',
  });

  // Model Shards (Shard A: 5,618,304 params; Shard B: 5,422,464 params; Total: 11,040,768 params)
  const [shardA, setShardA] = useState<ShardAWeights>(() => createShardAWeights(MODEL_SPEC.seed));
  const [shardB, setShardB] = useState<ShardBWeights>(() => createShardBWeights(MODEL_SPEC.seed));
  const [weightsSourceLabel, setWeightsSourceLabel] = useState<string>(
    'PyTorch torch.manual_seed(42) (Built-in FP32)'
  );

  // Inference Configuration & Prompt State
  const [promptText, setPromptText] = useState<string>('Hello');
  const [promptError, setPromptError] = useState<string | null>(null);
  const [maxNewBytes, setMaxNewBytes] = useState<number>(16);
  const [temperature, setTemperature] = useState<number>(0.0);
  const [transferMode, setTransferMode] = useState<'kv-step' | 'full-sequence' | 'raw-1byte'>('kv-step');

  // Live Generation Telemetry State
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [activeStage, setActiveStage] = useState<
    'idle' | 'node-a' | 'datachannel-tx' | 'node-b' | 'datachannel-rx'
  >('idle');
  const [promptByteList, setPromptByteList] = useState<number[]>([72, 101, 108, 108, 111]);
  const [generatedHistory, setGeneratedHistory] = useState<GeneratedByteRecord[]>([]);
  const [currentPos, setCurrentPos] = useState<number>(5);
  const [lastInputByte, setLastInputByte] = useState<number | null>(111);
  const [lastOutputByte, setLastOutputByte] = useState<number | null>(null);
  const [lastActivation, setLastActivation] = useState<Float32Array | null>(null);
  const [topKLogits, setTopKLogits] = useState<Array<{ byte: number; logit: number; prob: number }>>([]);
  const [nodeAComputeMs, setNodeAComputeMs] = useState<number>(0);
  const [nodeBComputeMs, setNodeBComputeMs] = useState<number>(0);
  const [throughputBps, setThroughputBps] = useState<number>(0);

  // Refs for real-time WebRTC callbacks and engines
  const meshRef = useRef<WebRTCMeshManager | null>(null);
  const nodeARuntimeRef = useRef<NodeARuntime>(new NodeARuntime(shardA));
  const nodeBRuntimeRef = useRef<NodeBRuntime>(new NodeBRuntime(shardB));
  const assemblerARef = useRef<DataChannelPacketAssembler>(new DataChannelPacketAssembler());
  const assemblerBRef = useRef<DataChannelPacketAssembler>(new DataChannelPacketAssembler());
  const samplingPrngRef = useRef<TorchMT19937>(new TorchMT19937(42));

  // Mutable generation state inside refs for asynchronous WebRTC ping-pong loop
  const genSessionRef = useRef<{
    active: boolean;
    sequence: number[];
    targetNewBytes: number;
    generatedCount: number;
    stepId: number;
    stepStartTimeMs: number;
    runStartTimeMs: number;
    lastNodeAMs: number;
    lastTxBytes: number;
    temperature: number;
    transferMode: 'kv-step' | 'full-sequence' | 'raw-1byte';
  }>({
    active: false,
    sequence: [],
    targetNewBytes: 16,
    generatedCount: 0,
    stepId: 0,
    stepStartTimeMs: 0,
    runStartTimeMs: 0,
    lastNodeAMs: 0,
    lastTxBytes: 1536,
    temperature: 0.0,
    transferMode: 'kv-step',
  });

  // Sync Shard A / Shard B updates to runtime instances
  useEffect(() => {
    nodeARuntimeRef.current = new NodeARuntime(shardA);
  }, [shardA]);

  useEffect(() => {
    nodeBRuntimeRef.current = new NodeBRuntime(shardB);
  }, [shardB]);

  // Initialize WebRTCMeshManager and bind Node A / Node B message handlers
  useEffect(() => {
    const mesh = new WebRTCMeshManager();
    meshRef.current = mesh;

    mesh.setCallbacks({
      onStateChange: (st, statsSnap) => {
        setConnectionState(st);
        setChannelStats(statsSnap);
      },
      // Node B (Phone 2) receives binary activation packet from Node A (Phone 1)
      onNodeBMessage: (buffer: ArrayBuffer) => {
        const pkt = assemblerBRef.current.decode(buffer);
        if (!pkt) return;

        if (pkt.type === PacketType.RESET_SESSION) {
          nodeBRuntimeRef.current.resetCache();
          if (!genSessionRef.current.active) {
            setGeneratedHistory([]);
          }
          return;
        }

        if (pkt.type === PacketType.ACTIVATION_STEP) {
          setActiveStage('node-b');
          const t0 = performance.now();
          const { logits } = nodeBRuntimeRef.current.forwardStep(pkt.activations, pkt.seqPos);
          const nodeBMs = performance.now() - t0;

          // stepId === 0 indicates prompt prefill for positions 0..promptLen-2
          if (pkt.stepId === 0) {
            setNodeBComputeMs(nodeBMs);
            setLastActivation(pkt.activations);
            setChannelStats({ ...mesh.stats });
            return;
          }

          const { nextByte, topK } = sampleNextByte(logits, pkt.temperature, samplingPrngRef.current);
          setNodeBComputeMs(nodeBMs);
          setTopKLogits(topK);
          setLastOutputByte(nextByte);
          setLastActivation(pkt.activations);

          if (!genSessionRef.current.active) {
            setCurrentPos(pkt.seqPos + 1);
            setGeneratedHistory((prev) => [
              ...prev,
              {
                step: prev.length + 1,
                seqPos: pkt.seqPos,
                byte: nextByte,
                nodeAMs: 0,
                nodeBMs,
                rttMs: nodeBMs,
                txBytes: pkt.activations.byteLength,
              },
            ]);
          }

          const raw1Byte = genSessionRef.current.transferMode === 'raw-1byte';
          const replyBuf = encodeTokenResultPacket(
            pkt.stepId,
            pkt.seqPos,
            nextByte,
            nodeBMs,
            topK,
            raw1Byte
          );
          setActiveStage('datachannel-rx');
          mesh.sendFromNodeB(replyBuf);
          setChannelStats({ ...mesh.stats });
        } else if (pkt.type === PacketType.ACTIVATION_CHUNK) {
          setActiveStage('node-b');
          const t0 = performance.now();
          const { logits } = nodeBRuntimeRef.current.forwardFullSequence(
            pkt.activations,
            pkt.seqLen
          );
          const nodeBMs = performance.now() - t0;

          const { nextByte, topK } = sampleNextByte(logits, pkt.temperature, samplingPrngRef.current);
          setNodeBComputeMs(nodeBMs);
          setTopKLogits(topK);
          setLastOutputByte(nextByte);
          setLastActivation(
            pkt.activations.subarray(
              Math.max(0, (pkt.seqLen - 1) * MODEL_SPEC.dModel),
              pkt.seqLen * MODEL_SPEC.dModel
            )
          );

          if (!genSessionRef.current.active) {
            setCurrentPos(pkt.seqLen);
            setGeneratedHistory((prev) => [
              ...prev,
              {
                step: prev.length + 1,
                seqPos: pkt.seqLen - 1,
                byte: nextByte,
                nodeAMs: 0,
                nodeBMs,
                rttMs: nodeBMs,
                txBytes: pkt.activations.byteLength,
              },
            ]);
          }

          const replyBuf = encodeTokenResultPacket(
            pkt.stepId,
            pkt.seqLen - 1,
            nextByte,
            nodeBMs,
            topK,
            false
          );
          setActiveStage('datachannel-rx');
          mesh.sendFromNodeB(replyBuf);
          setChannelStats({ ...mesh.stats });
        }
      },
      // Node A (Phone 1) receives 1-byte / token result packet back from Node B (Phone 2)
      onNodeAMessage: (buffer: ArrayBuffer) => {
        const pkt = assemblerARef.current.decode(buffer);
        if (!pkt || pkt.type !== PacketType.TOKEN_RESULT) return;

        const sess = genSessionRef.current;
        const now = performance.now();
        const rttMs = Math.max(0.1, now - sess.stepStartTimeMs);

        if (pkt.topK && pkt.topK.length > 0) {
          setTopKLogits(pkt.topK);
        }
        if (pkt.nodeBComputeMs > 0) {
          setNodeBComputeMs(pkt.nodeBComputeMs);
        }

        const nextByte = pkt.nextByte & 0xff;
        setLastOutputByte(nextByte);

        if (!sess.active) {
          setActiveStage('idle');
          return;
        }

        sess.sequence.push(nextByte);
        sess.generatedCount++;
        const newPos = sess.sequence.length - 1;
        setCurrentPos(sess.sequence.length);

        const elapsedTotalSec = Math.max(0.001, (now - sess.runStartTimeMs) / 1000);
        setThroughputBps(sess.generatedCount / elapsedTotalSec);

        setGeneratedHistory((prev) => [
          ...prev,
          {
            step: sess.generatedCount,
            seqPos: newPos,
            byte: nextByte,
            nodeAMs: sess.lastNodeAMs,
            nodeBMs: pkt.nodeBComputeMs,
            rttMs,
            txBytes: sess.lastTxBytes,
          },
        ]);
        setChannelStats({ ...mesh.stats, lastRttMs: rttMs });

        if (
          sess.generatedCount >= sess.targetNewBytes ||
          sess.sequence.length >= MODEL_SPEC.maxSeqLen
        ) {
          sess.active = false;
          setIsGenerating(false);
          setActiveStage('idle');
          return;
        }

        // Schedule next token's forward pass on Phone 1 (Node A)
        setTimeout(() => {
          if (!genSessionRef.current.active) return;
          executeNextNodeAStep();
        }, 12);
      },
    });

    // Establish initial Dual-Node WebRTC Loopback connection automatically
    mesh.connectDualNodeLoopback().catch(() => {});

    return () => {
      mesh.disconnect();
    };
  }, []);

  /**
   * Execute Phone 1 (Node A: Embeddings + Layers 0-2) for the newest byte in `genSessionRef.current.sequence`
   * and transmit the intermediate activation tensor over the WebRTC DataChannel to Phone 2.
   */
  const executeNextNodeAStep = () => {
    const mesh = meshRef.current;
    const sess = genSessionRef.current;
    if (!mesh || !sess.active) return;

    const seq = sess.sequence;
    const pos = seq.length - 1;
    const currentByte = seq[pos];
    setLastInputByte(currentByte);
    setActiveStage('node-a');

    sess.stepId++;
    sess.stepStartTimeMs = performance.now();

    if (sess.transferMode === 'full-sequence') {
      const t0 = performance.now();
      const fullAct = nodeARuntimeRef.current.forwardFullSequence(seq, false);
      const nodeAMs = performance.now() - t0;
      sess.lastNodeAMs = nodeAMs;
      setNodeAComputeMs(nodeAMs);

      // Extract the last token's 384-d slice for the live heatmap
      const lastSlice = fullAct.slice((seq.length - 1) * MODEL_SPEC.dModel, seq.length * MODEL_SPEC.dModel);
      setLastActivation(lastSlice);

      const chunks = encodeActivationFullChunks(
        sess.stepId,
        seq.length,
        sess.temperature,
        fullAct
      );
      sess.lastTxBytes = fullAct.byteLength;
      setActiveStage('datachannel-tx');
      for (const ch of chunks) {
        mesh.sendFromNodeA(ch);
      }
      setChannelStats({ ...mesh.stats });
    } else {
      const t0 = performance.now();
      const stepAct = nodeARuntimeRef.current.forwardStep(currentByte, pos);
      const nodeAMs = performance.now() - t0;
      sess.lastNodeAMs = nodeAMs;
      setNodeAComputeMs(nodeAMs);
      setLastActivation(stepAct);

      const pkt = encodeActivationStepPacket(sess.stepId, pos, sess.temperature, stepAct);
      sess.lastTxBytes = stepAct.byteLength;
      setActiveStage('datachannel-tx');
      mesh.sendFromNodeA(pkt);
      setChannelStats({ ...mesh.stats });
    }
  };

  /**
   * Start distributed split inference across Phone 1 (Layers 0-2) and Phone 2 (Layers 3-5)
   */
  const handleStartSplitInference = async () => {
    if (isGenerating) {
      genSessionRef.current.active = false;
      setIsGenerating(false);
      setActiveStage('idle');
      return;
    }

    if (!promptText || promptText.length === 0) {
      setPromptError('Enter at least 1 character or byte in the prompt input before running split inference.');
      return;
    }
    setPromptError(null);

    const mesh = meshRef.current;
    if (!mesh) return;

    if (connectionState !== 'connected') {
      if (role === 'dual-loopback') {
        await mesh.connectDualNodeLoopback();
      } else {
        setActiveTab('pairing');
        return;
      }
    }

    const encoder = new TextEncoder();
    const initialBytes = Array.from(encoder.encode(promptText)).slice(
      0,
      MODEL_SPEC.maxSeqLen - maxNewBytes
    );

    setPromptByteList(initialBytes);
    setGeneratedHistory([]);
    setCurrentPos(initialBytes.length);
    setThroughputBps(0);

    // Reset KV caches on both Node A and Node B
    nodeARuntimeRef.current.resetCache();
    nodeBRuntimeRef.current.resetCache();
    samplingPrngRef.current.manualSeed(MODEL_SPEC.seed);

    mesh.sendFromNodeA(encodeResetSessionPacket(Date.now() & 0xffffffff));

    // Stream prefill prompt bytes 0..initialBytes.length - 2 over WebRTC DataChannel (stepId = 0)
    if (transferMode !== 'full-sequence' && initialBytes.length > 1) {
      for (let i = 0; i < initialBytes.length - 1; i++) {
        const act = nodeARuntimeRef.current.forwardStep(initialBytes[i], i);
        mesh.sendFromNodeA(encodeActivationStepPacket(0, i, temperature, act));
      }
    }

    genSessionRef.current = {
      active: true,
      sequence: [...initialBytes],
      targetNewBytes: maxNewBytes,
      generatedCount: 0,
      stepId: 0,
      stepStartTimeMs: performance.now(),
      runStartTimeMs: performance.now(),
      lastNodeAMs: 0,
      lastTxBytes: 1536,
      temperature,
      transferMode,
    };

    setIsGenerating(true);
    executeNextNodeAStep();
  };

  const handleSelectRole = async (newRole: PeerRole) => {
    genSessionRef.current.active = false;
    setIsGenerating(false);
    setRole(newRole);
    if (newRole === 'dual-loopback' && meshRef.current) {
      await meshRef.current.connectDualNodeLoopback();
    } else if (meshRef.current) {
      meshRef.current.disconnect();
    }
  };

  const handleImportWeightsBuffer = (buffer: ArrayBuffer, fileName: string) => {
    const floatView = new Float32Array(buffer);
    if (floatView.length >= MODEL_SPEC.totalParams) {
      const bufA = new Float32Array(floatView.subarray(0, MODEL_SPEC.shardAParams));
      const bufB = new Float32Array(
        floatView.subarray(MODEL_SPEC.shardAParams, MODEL_SPEC.totalParams)
      );
      setShardA(createShardAWeights(MODEL_SPEC.seed, bufA));
      setShardB(createShardBWeights(MODEL_SPEC.seed, bufB));
      setWeightsSourceLabel(`${fileName} (Full 44.16 MB FP32 Split)`);
    } else if (floatView.length === MODEL_SPEC.shardAParams) {
      setShardA(createShardAWeights(MODEL_SPEC.seed, new Float32Array(floatView)));
      setWeightsSourceLabel(`${fileName} (Shard A 22.47 MB FP32)`);
    } else if (floatView.length === MODEL_SPEC.shardBParams) {
      setShardB(createShardBWeights(MODEL_SPEC.seed, new Float32Array(floatView)));
      setWeightsSourceLabel(`${fileName} (Shard B 21.69 MB FP32)`);
    }
  };

  const handleResetSeed42Weights = () => {
    setShardA(createShardAWeights(MODEL_SPEC.seed));
    setShardB(createShardBWeights(MODEL_SPEC.seed));
    setWeightsSourceLabel('PyTorch torch.manual_seed(42) (Built-in FP32)');
  };

  const allSequenceBytes = [
    ...promptByteList,
    ...generatedHistory.map((item) => item.byte),
  ];

  return (
    <div className="min-h-screen bg-[#0B0E17] text-slate-100 flex flex-col">
      {/* TOP BAR CONTRACT: Zone 1 (Brand) — Zone 2 (4 Nav Links) — Zone 3 (1 Primary CTA) */}
      <header className="flex flex-wrap md:flex-nowrap items-center justify-between gap-y-2.5 gap-x-8 px-4 md:px-6 py-3 border-b border-slate-800 bg-[#0B0E17]">
        <a
          href="#workspace"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('workspace');
          }}
          className="text-lg font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
        >
          SplitByte
        </a>

        <nav
          aria-label="Primary Navigation"
          className="order-3 md:order-2 w-full md:w-auto flex flex-wrap items-center justify-start md:justify-center gap-x-4 gap-y-1.5 md:gap-6 text-xs md:text-sm font-medium text-slate-400 border-t md:border-t-0 border-slate-800/80 pt-2 md:pt-0"
        >
          <button
            type="button"
            onClick={() => setActiveTab('workspace')}
            className={`py-1 border-b-2 transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'workspace'
                ? 'border-cyan-400 text-slate-100'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Inference Console
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('pairing')}
            className={`py-1 border-b-2 transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'pairing'
                ? 'border-cyan-400 text-slate-100'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            P2P Pairing
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('architecture')}
            className={`py-1 border-b-2 transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'architecture'
                ? 'border-cyan-400 text-slate-100'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Architecture &amp; Weights
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('determinism')}
            className={`py-1 border-b-2 transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'determinism'
                ? 'border-cyan-400 text-slate-100'
                : 'border-transparent hover:text-slate-200'
            }`}
          >
            Determinism Suite
          </button>
        </nav>

        <div className="order-2 md:order-3 flex items-center shrink-0">
          <button
            type="button"
            onClick={handleStartSplitInference}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 ${
              isGenerating
                ? 'bg-rose-500 hover:bg-rose-400 text-slate-950'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
            }`}
          >
            {isGenerating ? 'Stop Split Inference' : 'Run Split Inference'}
          </button>
        </div>
      </header>

      {/* Sub-Header Telemetry Ribbon (36px Science/Telemetry Workbench Bar) */}
      <div className="px-6 py-2 bg-[#111827] border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-4 text-xs font-mono tabular-nums text-slate-400">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-slate-200">
            Architecture: 6-Layer Causal GPT · Vocab: 256 Bytes · d_model: 384 · Heads: 6 · d_ff: 1536
          </span>
          <span aria-hidden="true">·</span>
          <span>Params: 11,040,768 (44.16 MB FP32)</span>
          <span aria-hidden="true">·</span>
          <span>Compute: Web CPU Only</span>
        </div>
        <div className="flex items-center gap-3">
          <span>
            Role:{' '}
            <strong className="text-cyan-300">
              {role === 'dual-loopback'
                ? 'Dual-Node WebRTC Loopback'
                : role === 'node-a'
                ? 'Phone 1 (Layers 0–2)'
                : 'Phone 2 (Layers 3–5)'}
            </strong>
          </span>
          <span aria-hidden="true">·</span>
          <span>
            Throughput: <strong className="text-emerald-400">{throughputBps.toFixed(1)} B/s</strong>
          </span>
        </div>
      </div>

      {/* Main Workspace Container */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-6">
        {activeTab === 'workspace' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* LEFT CONTROL & PARAMETER COLUMN (4 cols on 12-col grid) */}
            <aside className="lg:col-span-4 bg-[#111827] border border-slate-800 rounded-lg p-5 space-y-5">
              <div className="border-b border-slate-800 pb-3">
                <h1 className="text-base font-semibold text-slate-100">
                  Split Inference Parameters
                </h1>
                <p className="text-xs text-slate-400 mt-0.5">
                  Configure byte prompt, deterministic temperature, and RTCDataChannel tensor framing
                </p>
              </div>

              {/* Prompt Input */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label htmlFor="prompt-byte-input" className="text-xs font-medium text-slate-200">
                    Input Byte Sequence (UTF-8 / ASCII)
                  </label>
                  <span className="text-[11px] font-mono text-slate-400 tabular-nums">
                    {new TextEncoder().encode(promptText).length} bytes
                  </span>
                </div>
                <textarea
                  id="prompt-byte-input"
                  rows={2}
                  value={promptText}
                  onChange={(e) => {
                    setPromptText(e.target.value);
                    if (promptError) setPromptError(null);
                  }}
                  placeholder="Enter raw ASCII/UTF-8 text prompt..."
                  className="w-full rounded bg-slate-950 border border-slate-800 p-2.5 text-xs font-mono text-slate-100 focus:border-cyan-400 focus:outline-none"
                />
                {promptError && (
                  <p role="alert" className="mt-1.5 text-xs font-mono text-rose-400">
                    ✖ {promptError}
                  </p>
                )}

                {/* Preset Prompt Buttons */}
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] text-slate-400 mr-1">Presets:</span>
                  {['Hello', 'The Transformer ', 'def forward(x):', '010101'].map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => {
                        setPromptText(preset);
                        setPromptError(null);
                      }}
                      className="px-2 py-1 rounded border border-slate-800 bg-slate-950 hover:border-slate-700 text-[11px] font-mono text-slate-300 transition-colors whitespace-nowrap shrink-0"
                    >
                      {preset.trim()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Sampling Temperature Control */}
              <div>
                <div className="flex items-center justify-between mb-1.5 text-xs">
                  <label htmlFor="temperature-slider" className="font-medium text-slate-200">
                    Sampling Temperature
                  </label>
                  <span className="font-mono text-cyan-300 tabular-nums">
                    {temperature.toFixed(2)} {temperature === 0 ? '(Deterministic Argmax)' : '(Stochastic)'}
                  </span>
                </div>
                <input
                  id="temperature-slider"
                  type="range"
                  min={0}
                  max={1.2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature( parseFloat(e.target.value) )}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
                <div className="flex justify-between text-[11px] font-mono text-slate-400 mt-1 tabular-nums">
                  <span>0.0 (Seed 42 Exact)</span>
                  <span>0.6</span>
                  <span>1.2</span>
                </div>
              </div>

              {/* Number of Bytes to Generate */}
              <div>
                <label htmlFor="max-bytes-select" className="block text-xs font-medium text-slate-200 mb-1.5">
                  Target Bytes to Generate
                </label>
                <div className="grid grid-cols-4 gap-1.5 p-1 bg-slate-950 rounded-lg border border-slate-800">
                  {[8, 16, 32, 64].map((count) => (
                    <button
                      key={count}
                      type="button"
                      onClick={() => setMaxNewBytes(count)}
                      className={`py-1.5 text-xs font-mono rounded transition-colors whitespace-nowrap shrink-0 tabular-nums ${
                        maxNewBytes === count
                          ? 'bg-cyan-500 text-slate-950 font-semibold'
                          : 'text-slate-300 hover:text-slate-100'
                      }`}
                    >
                      {count} B
                    </button>
                  ))}
                </div>
              </div>

              {/* DataChannel Tensor Framing Selector */}
              <div>
                <label htmlFor="transfer-mode-select" className="block text-xs font-medium text-slate-200 mb-1.5">
                  RTCDataChannel Activation Protocol
                </label>
                <select
                  id="transfer-mode-select"
                  value={transferMode}
                  onChange={(e) =>
                    setTransferMode(e.target.value as 'kv-step' | 'full-sequence' | 'raw-1byte')
                  }
                  className="w-full px-3 py-2 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:border-cyan-400 focus:outline-none"
                >
                  <option value="kv-step">KV-Cached Activation [1 × 384 Float32] (1.5 KB/step)</option>
                  <option value="full-sequence">Full Sequence Tensor [1 × T × 384 Float32] (SCTP Chunked)</option>
                  <option value="raw-1byte">Strict 1-Byte Uint8Array([nextByte]) Reply Mode</option>
                </select>
              </div>

              {/* Primary Execution Trigger */}
              <div className="pt-2 border-t border-slate-800 space-y-2">
                <button
                  type="button"
                  onClick={handleStartSplitInference}
                  className={`w-full py-2.5 px-4 rounded-lg font-semibold text-xs transition-colors whitespace-nowrap shrink-0 ${
                    isGenerating
                      ? 'bg-rose-500 hover:bg-rose-400 text-slate-950'
                      : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
                  }`}
                >
                  {isGenerating
                    ? `Stop Generation (${generatedHistory.length}/${maxNewBytes} Bytes)`
                    : `Execute P2P Split Pass (${maxNewBytes} Bytes)`}
                </button>

                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 tabular-nums">
                  <span>Last RTT: {channelStats.lastRttMs.toFixed(2)} ms</span>
                  <span>
                    Total Transferred: {((channelStats.bytesSent + channelStats.bytesReceived) / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>

              {/* Live Byte Stream Output Card */}
              <div className="pt-3 border-t border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <h2 className="text-xs font-semibold text-slate-200">
                    Generated Byte Sequence ({generatedHistory.length} / {maxNewBytes} bytes)
                  </h2>
                  <span className="text-[11px] font-mono text-cyan-400 tabular-nums">
                    Seq Len: {allSequenceBytes.length} / 512
                  </span>
                </div>

                {/* Decoded Glyph / Byte Stream Box */}
                <div
                  className="p-3 rounded bg-slate-950 border border-slate-800 font-mono text-xs min-h-[76px] break-all leading-relaxed"
                  data-testid="generated-byte-stream"
                >
                  <span className="text-slate-300" title="Input Prompt Bytes">
                    {promptByteList.map((b) => formatByteGlyph(b)).join('')}
                  </span>
                  <span className="text-cyan-300 font-semibold" title="Generated Bytes from P2P Split Inference">
                    {generatedHistory.map((item) => formatByteGlyph(item.byte)).join('')}
                  </span>
                  {generatedHistory.length === 0 && !isGenerating && (
                    <span className="text-slate-500 ml-1">
                      [Click &quot;Run Split Inference&quot; to stream bytes]
                    </span>
                  )}
                </div>

                {/* Hexadecimal Byte Tape */}
                {generatedHistory.length > 0 && (
                  <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800/80 text-[11px] font-mono text-amber-300 tabular-nums overflow-x-auto whitespace-nowrap">
                    HEX:{' '}
                    {generatedHistory
                      .map((item) => '0x' + item.byte.toString(16).toUpperCase().padStart(2, '0'))
                      .join(' ')}
                  </div>
                )}
              </div>
            </aside>

            {/* MAIN VIEWPORT / PIPELINE TELEMETRY STAGE (8 cols on 12-col grid) */}
            <section className="lg:col-span-8 space-y-6">
              <PipelineVisualizer
                role={role}
                connectionState={connectionState}
                isGenerating={isGenerating}
                activeStage={activeStage}
                currentPos={allSequenceBytes.length}
                lastInputByte={lastInputByte}
                lastOutputByte={lastOutputByte}
                lastActivation={lastActivation}
                topK={topKLogits}
                nodeAComputeMs={nodeAComputeMs}
                nodeBComputeMs={nodeBComputeMs}
                stats={channelStats}
                shardAChecksum={shardA.checksum}
                shardBChecksum={shardB.checksum}
                transferMode={transferMode}
              />

              {/* Per-Byte P2P Packet Telemetry Log Table */}
              <div className="bg-[#111827] border border-slate-800 rounded-lg p-5">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      Per-Token Distributed Execution Ledger
                    </h3>
                    <p className="text-xs text-slate-400">
                      Step-by-step latency across Phone 1 (Layers 0–2), WebRTC DataChannel, and Phone 2 (Layers 3–5)
                    </p>
                  </div>
                  <span className="text-xs font-mono text-slate-400 tabular-nums">
                    {generatedHistory.length} steps logged
                  </span>
                </div>

                {generatedHistory.length === 0 ? (
                  <div className="py-8 text-center border border-dashed border-slate-800 rounded text-xs font-mono text-slate-400">
                    No tokens generated in current session yet. Press &quot;Run Split Inference&quot; to begin P2P pipeline execution.
                  </div>
                ) : (
                  <div className="overflow-x-auto max-h-56">
                    <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400">
                          <th className="py-2 px-2.5">Step</th>
                          <th className="py-2 px-2.5">Pos</th>
                          <th className="py-2 px-2.5">Output Byte (Dec / Hex / Glyph)</th>
                          <th className="py-2 px-2.5 text-right">Node A (L0–L2)</th>
                          <th className="py-2 px-2.5 text-right">DataChannel TX</th>
                          <th className="py-2 px-2.5 text-right">Node B (L3–L5)</th>
                          <th className="py-2 px-2.5 text-right">Step Total</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/70 text-slate-200">
                        {generatedHistory.map((item) => (
                          <tr key={item.step} className="hover:bg-slate-900/50">
                            <td className="py-1.5 px-2.5 text-slate-400">#{item.step}</td>
                            <td className="py-1.5 px-2.5">{item.seqPos}</td>
                            <td className="py-1.5 px-2.5 text-cyan-300">
                              {item.byte} · 0x{item.byte.toString(16).toUpperCase().padStart(2, '0')} ·{' '}
                              {formatByteGlyph(item.byte)}
                            </td>
                            <td className="py-1.5 px-2.5 text-right">{item.nodeAMs.toFixed(2)} ms</td>
                            <td className="py-1.5 px-2.5 text-right text-slate-400">
                              {item.txBytes.toLocaleString()} B
                            </td>
                            <td className="py-1.5 px-2.5 text-right text-amber-300">
                              {item.nodeBMs.toFixed(2)} ms
                            </td>
                            <td className="py-1.5 px-2.5 text-right text-emerald-400">
                              {item.rttMs.toFixed(2)} ms
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}

        {activeTab === 'pairing' && (
          <PairingPanel
            role={role}
            connectionState={connectionState}
            stats={channelStats}
            errorMessage={meshRef.current?.errorMessage ?? null}
            onSelectRole={handleSelectRole}
            onConnectLoopback={async () => {
              if (meshRef.current) await meshRef.current.connectDualNodeLoopback();
            }}
            onStartBroadcastAutoPair={async (r, code) => {
              if (meshRef.current) await meshRef.current.startBroadcastAutoPair(r, code);
            }}
            onCreatePhone1Offer={async () => {
              if (!meshRef.current) throw new Error('Mesh manager not initialized');
              return await meshRef.current.createPhone1Offer();
            }}
            onAcceptOfferAndCreatePhone2Answer={async (offerToken) => {
              if (!meshRef.current) throw new Error('Mesh manager not initialized');
              return await meshRef.current.acceptOfferAndCreatePhone2Answer(offerToken);
            }}
            onCompletePhone1Handshake={async (answerToken) => {
              if (!meshRef.current) throw new Error('Mesh manager not initialized');
              await meshRef.current.completePhone1Handshake(answerToken);
            }}
            onDisconnect={() => {
              meshRef.current?.disconnect();
            }}
          />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureSpecPanel
            shardA={shardA}
            shardB={shardB}
            weightsSourceLabel={weightsSourceLabel}
            onImportWeightsBuffer={handleImportWeightsBuffer}
            onResetSeed42Weights={handleResetSeed42Weights}
          />
        )}

        {activeTab === 'determinism' && (
          <DeterminismSuitePanel shardA={shardA} shardB={shardB} />
        )}
      </main>
    </div>
  );
}
