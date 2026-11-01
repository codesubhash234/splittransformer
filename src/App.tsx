import React, { useEffect, useRef, useState } from 'react';
import {
  ComputeBackend,
  MODEL_SPEC,
  NodeARuntime,
  NodeBRuntime,
  ShardAWeights,
  ShardBWeights,
  WeightInitScheme,
  createShardAWeights,
  createShardBWeights,
  sampleNextByte,
  formatByteGlyph,
} from './engine/transformer';
import {
  DataChannelPacketAssembler,
  PacketType,
  encodeActivationFullChunks,
  encodeActivationStepInt8Packet,
  encodeActivationStepPacket,
  encodePrefillBatchPacket,
  encodeRemoteStartRequestPacket,
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
  inputByte: number;
  byte: number;
  nodeAMs: number;
  nodeBMs: number;
  rttMs: number;
  txBytes: number;
}

export default function App() {
  const hasOfferHash =
    typeof window !== 'undefined' && window.location.hash.startsWith('#offer=');
  const [activeTab, setActiveTab] = useState<ActiveTab>(
    hasOfferHash ? 'pairing' : 'workspace'
  );
  const [role, setRole] = useState<PeerRole>(
    hasOfferHash ? 'node-b' : 'dual-loopback'
  );
  const [quickRoomCode, setQuickRoomCode] = useState<string>('4242');
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [channelStats, setChannelStats] = useState<DataChannelStats>({
    messagesSent: 0,
    messagesReceived: 0,
    bytesSent: 0,
    bytesReceived: 0,
    lastPacketBytes: 0,
    lastRttMs: 0,
    transportMode: 'p2p-webrtc-loopback',
    iceCandidateType: 'Single-Device Loopback (pcA ⇄ pcB)',
  });

  // Model Shards (Shard A: 5,618,304 params; Shard B: 5,422,464 params; Total: 11,040,768 params)
  const [initScheme, setInitScheme] = useState<WeightInitScheme>('box-muller-seed42');
  const [computeBackend, setComputeBackend] = useState<ComputeBackend>('wasm-cpu');
  const [shardA, setShardA] = useState<ShardAWeights>(() =>
    createShardAWeights(MODEL_SPEC.seed, undefined, 'box-muller-seed42')
  );
  const [shardB, setShardB] = useState<ShardBWeights>(() =>
    createShardBWeights(MODEL_SPEC.seed, undefined, 'box-muller-seed42')
  );
  const [weightsSourceLabel, setWeightsSourceLabel] = useState<string>(
    'PyTorch torch.manual_seed(42) (True Box-Muller FP32)'
  );

  // Inference Configuration & Prompt State
  const [promptText, setPromptText] = useState<string>('Hello');
  const [promptError, setPromptError] = useState<string | null>(null);
  const [maxNewBytes, setMaxNewBytes] = useState<number>(16);
  const [temperature, setTemperature] = useState<number>(0.0);
  const [transferMode, setTransferMode] = useState<
    'kv-step' | 'kv-int8' | 'full-sequence' | 'raw-1byte'
  >('kv-step');
  const [pacingMode, setPacingMode] = useState<'turbo-0ms' | 'visual-60ms'>('turbo-0ms');
  const [speculativeOverlap, setSpeculativeOverlap] = useState<boolean>(true);

  // Live Generation Telemetry State (Synchronized across BOTH Phone 1 and Phone 2)
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
  const [ttftMs, setTtftMs] = useState<number>(0);
  const [speculativeHits, setSpeculativeHits] = useState<number>(0);

  // Refs for real-time WebRTC callbacks and engines
  const meshRef = useRef<WebRTCMeshManager | null>(null);
  const roleRef = useRef<PeerRole>(role);
  const pacingModeRef = useRef<'turbo-0ms' | 'visual-60ms'>(pacingMode);
  const speculativeOverlapRef = useRef<boolean>(speculativeOverlap);
  const specCacheRef = useRef<{
    pos: number;
    predictedByte: number;
    activation: Float32Array;
  } | null>(null);
  const nodeARuntimeRef = useRef<NodeARuntime>(new NodeARuntime(shardA));
  const nodeBRuntimeRef = useRef<NodeBRuntime>(new NodeBRuntime(shardB));
  const assemblerARef = useRef<DataChannelPacketAssembler>(new DataChannelPacketAssembler());
  const assemblerBRef = useRef<DataChannelPacketAssembler>(new DataChannelPacketAssembler());
  const samplingPrngRef = useRef<TorchMT19937>(new TorchMT19937(42));
  const startTriggerRef = useRef<() => void>(() => {});

  useEffect(() => {
    roleRef.current = role;
  }, [role]);

  useEffect(() => {
    pacingModeRef.current = pacingMode;
  }, [pacingMode]);

  useEffect(() => {
    speculativeOverlapRef.current = speculativeOverlap;
  }, [speculativeOverlap]);

  const genSessionRef = useRef<{
    active: boolean;
    sequence: number[];
    targetNewBytes: number;
    generatedCount: number;
    stepId: number;
    stepStartTimeMs: number;
    runStartTimeMs: number;
    lastNodeAMs: number;
    lastInputByte: number;
    lastTxBytes: number;
    lastActivationVec?: Float32Array | null;
    temperature: number;
    transferMode: 'kv-step' | 'kv-int8' | 'full-sequence' | 'raw-1byte';
  }>({
    active: false,
    sequence: [],
    targetNewBytes: 16,
    generatedCount: 0,
    stepId: 0,
    stepStartTimeMs: 0,
    runStartTimeMs: 0,
    lastNodeAMs: 0,
    lastInputByte: 111,
    lastTxBytes: 1536,
    lastActivationVec: null,
    temperature: 0.0,
    transferMode: 'kv-step',
  });

  useEffect(() => {
    nodeARuntimeRef.current = new NodeARuntime(shardA, computeBackend);
  }, [shardA, computeBackend]);

  useEffect(() => {
    nodeBRuntimeRef.current = new NodeBRuntime(shardB, computeBackend);
  }, [shardB, computeBackend]);

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
          samplingPrngRef.current.manualSeed(MODEL_SPEC.seed);
          if (!genSessionRef.current.active) {
            setGeneratedHistory([]);
            setIsGenerating(true);
            setActiveStage('datachannel-tx');
            if (pkt.promptBytes && pkt.promptBytes.length > 0) {
              setPromptByteList(pkt.promptBytes);
              setCurrentPos(pkt.promptBytes.length);
              setLastInputByte(pkt.promptBytes[pkt.promptBytes.length - 1]);
            }
            if (pkt.targetNewBytes) {
              setMaxNewBytes(pkt.targetNewBytes);
              genSessionRef.current.targetNewBytes = pkt.targetNewBytes;
            }
            genSessionRef.current.runStartTimeMs = performance.now();
          }
          return;
        }

        if (pkt.type === PacketType.PREFILL_BATCH) {
          const t0 = performance.now();
          const D = MODEL_SPEC.dModel;
          for (let i = 0; i < pkt.count; i++) {
            const slice = pkt.activations.subarray(i * D, (i + 1) * D);
            nodeBRuntimeRef.current.forwardStep(slice, i);
          }
          const nodeBMs = performance.now() - t0;
          if (!genSessionRef.current.active) {
            setNodeBComputeMs(nodeBMs);
            setChannelStats({ ...mesh.stats });
          }
          return;
        }

        if (pkt.type === PacketType.ACTIVATION_STEP) {
          const t0 = performance.now();
          const { logits } = nodeBRuntimeRef.current.forwardStep(pkt.activations, pkt.seqPos);
          const nodeBMs = performance.now() - t0;

          // stepId === 0 indicates prompt prefill for positions 0..promptLen-2
          if (pkt.stepId === 0) {
            if (!genSessionRef.current.active) {
              setNodeBComputeMs(nodeBMs);
              setLastActivation(pkt.activations);
              if (pkt.inputByte) setLastInputByte(pkt.inputByte);
              setChannelStats({ ...mesh.stats });
            }
            return;
          }

          const { nextByte, topK } = sampleNextByte(logits, pkt.temperature, samplingPrngRef.current);

          const raw1Byte = genSessionRef.current.transferMode === 'raw-1byte';
          const replyBuf = encodeTokenResultPacket(
            pkt.stepId,
            pkt.seqPos,
            nextByte,
            nodeBMs,
            topK,
            raw1Byte
          );
          mesh.sendFromNodeB(replyBuf);

          // Only update React UI from onNodeBMessage when running as standalone Phone 2 (Node B)
          if (!genSessionRef.current.active) {
            const stepNum = pkt.stepId;
            const isFinal = stepNum >= genSessionRef.current.targetNewBytes;
            const shouldRender =
              pacingModeRef.current !== 'turbo-0ms' || stepNum === 1 || stepNum % 4 === 0 || isFinal;

            if (shouldRender) {
              setActiveStage(isFinal ? 'idle' : 'node-b');
              setNodeBComputeMs(nodeBMs);
              setTopKLogits(topK);
              setLastOutputByte(nextByte);
              setLastActivation(pkt.activations);
              if (pkt.inputByte) setLastInputByte(pkt.inputByte);
              const elapsedSec = Math.max(
                0.001,
                (performance.now() - (genSessionRef.current.runStartTimeMs || performance.now())) / 1000
              );
              setThroughputBps(stepNum / elapsedSec);
              setCurrentPos(pkt.seqPos + 2);
              setChannelStats({ ...mesh.stats });
            }

            setGeneratedHistory((prev) => {
              if (prev.some((item) => item.step === stepNum)) return prev;
              const updated = [
                ...prev,
                {
                  step: stepNum,
                  seqPos: pkt.seqPos + 1,
                  inputByte: pkt.inputByte,
                  byte: nextByte,
                  nodeAMs: 0,
                  nodeBMs,
                  rttMs: nodeBMs,
                  txBytes: pkt.wireBytes || pkt.activations.byteLength,
                },
              ];
              if (updated.length >= genSessionRef.current.targetNewBytes) {
                setIsGenerating(false);
                setActiveStage('idle');
              }
              return updated;
            });
          }
        } else if (pkt.type === PacketType.ACTIVATION_CHUNK) {
          setActiveStage('node-b');
          const t0 = performance.now();
          const { logits } = nodeBRuntimeRef.current.forwardFullSequence(
            pkt.activations,
            pkt.seqLen
          );
          const nodeBMs = performance.now() - t0;

          const { nextByte, topK } = sampleNextByte(logits, pkt.temperature, samplingPrngRef.current);

          const replyBuf = encodeTokenResultPacket(
            pkt.stepId,
            pkt.seqLen - 1,
            nextByte,
            nodeBMs,
            topK,
            false
          );
          mesh.sendFromNodeB(replyBuf);

          if (!genSessionRef.current.active) {
            setNodeBComputeMs(nodeBMs);
            setTopKLogits(topK);
            setLastOutputByte(nextByte);
            if (pkt.inputByte) setLastInputByte(pkt.inputByte);
            setLastActivation(
              pkt.activations.subarray(
                Math.max(0, (pkt.seqLen - 1) * MODEL_SPEC.dModel),
                pkt.seqLen * MODEL_SPEC.dModel
              )
            );
            setCurrentPos(pkt.seqLen + 1);
            setChannelStats({ ...mesh.stats });
            setGeneratedHistory((prev) => {
              const updated = [
                ...prev,
                {
                  step: prev.length + 1,
                  seqPos: pkt.seqLen,
                  inputByte: pkt.inputByte,
                  byte: nextByte,
                  nodeAMs: 0,
                  nodeBMs,
                  rttMs: nodeBMs,
                  txBytes: pkt.activations.byteLength,
                },
              ];
              if (updated.length >= genSessionRef.current.targetNewBytes) {
                setIsGenerating(false);
                setActiveStage('idle');
              }
              return updated;
            });
          }
        }
      },
      // Node A (Phone 1) receives 1-byte / token result packet (or Remote Start Request) from Node B (Phone 2)
      onNodeAMessage: (buffer: ArrayBuffer) => {
        const pkt = assemblerARef.current.decode(buffer);
        if (!pkt) return;

        if (pkt.type === PacketType.REMOTE_START_REQUEST) {
          if (!genSessionRef.current.active) {
            startTriggerRef.current();
          }
          return;
        }

        if (pkt.type !== PacketType.TOKEN_RESULT) return;

        const sess = genSessionRef.current;
        if (pkt.stepId > 0 && pkt.stepId !== sess.stepId) {
          return;
        }
        const now = performance.now();
        const rttMs = Math.max(0.1, now - sess.stepStartTimeMs);
        const nextByte = pkt.nextByte & 0xff;

        if (!sess.active) {
          setActiveStage('idle');
          return;
        }

        sess.sequence.push(nextByte);
        sess.generatedCount++;
        const currentStepNum = sess.generatedCount;
        const newPos = sess.sequence.length - 1;
        const stepInputByte = sess.lastInputByte;
        const stepNodeAMs = sess.lastNodeAMs;
        const stepTxBytes = sess.lastTxBytes;
        const isFinalStep =
          sess.generatedCount >= sess.targetNewBytes ||
          sess.sequence.length >= MODEL_SPEC.maxSeqLen;

        if (currentStepNum === 1) {
          setTtftMs(Math.max(0.5, now - sess.runStartTimeMs));
        }

        const shouldUpdateHeavyUi =
          pacingModeRef.current !== 'turbo-0ms' ||
          currentStepNum === 1 ||
          currentStepNum % 4 === 0 ||
          isFinalStep;

        if (shouldUpdateHeavyUi) {
          if (pkt.topK && pkt.topK.length > 0) {
            setTopKLogits(pkt.topK);
          }
          if (pkt.nodeBComputeMs > 0) {
            setNodeBComputeMs(pkt.nodeBComputeMs);
          }
          setLastOutputByte(nextByte);
          setLastInputByte(stepInputByte);
          setNodeAComputeMs(stepNodeAMs);
          if (sess.lastActivationVec) {
            setLastActivation(sess.lastActivationVec);
          }
          setCurrentPos(sess.sequence.length);
          const elapsedTotalSec = Math.max(0.001, (now - sess.runStartTimeMs) / 1000);
          setThroughputBps(currentStepNum / elapsedTotalSec);
          setChannelStats({ ...mesh.stats, lastRttMs: rttMs });
        }

        setGeneratedHistory((prev) => {
          if (prev.some((item) => item.step === currentStepNum)) return prev;
          return [
            ...prev,
            {
              step: currentStepNum,
              seqPos: newPos,
              inputByte: stepInputByte,
              byte: nextByte,
              nodeAMs: stepNodeAMs,
              nodeBMs: pkt.nodeBComputeMs,
              rttMs,
              txBytes: stepTxBytes,
            },
          ];
        });

        if (isFinalStep) {
          sess.active = false;
          setIsGenerating(false);
          setActiveStage('idle');
          return;
        }

        if (pacingModeRef.current === 'turbo-0ms') {
          queueMicrotask(() => {
            if (!genSessionRef.current.active) return;
            executeNextNodeAStep(pkt.topK?.[0]?.byte);
          });
        } else {
          const stepDelayMs = roleRef.current === 'dual-loopback' ? 30 : 60;
          setTimeout(() => {
            if (!genSessionRef.current.active) return;
            executeNextNodeAStep(pkt.topK?.[0]?.byte);
          }, stepDelayMs);
        }
      },
    });

    if (!hasOfferHash) {
      mesh.connectDualNodeLoopback().catch(() => {});
    }

    return () => {
      mesh.disconnect();
    };
  }, []);

  const executeNextNodeAStep = (hintCandidateByte?: number) => {
    const mesh = meshRef.current;
    const sess = genSessionRef.current;
    if (!mesh || !sess.active) return;

    const seq = sess.sequence;
    const pos = seq.length - 1;
    const currentByte = seq[pos];
    sess.lastInputByte = currentByte;
    if (pacingModeRef.current !== 'turbo-0ms') {
      setLastInputByte(currentByte);
      setActiveStage('node-a');
    }

    sess.stepId++;
    sess.stepStartTimeMs = performance.now();

    if (sess.transferMode === 'full-sequence') {
      const t0 = performance.now();
      const fullAct = nodeARuntimeRef.current.forwardFullSequence(seq, false);
      const nodeAMs = performance.now() - t0;
      sess.lastNodeAMs = nodeAMs;
      const lastSlice = fullAct.slice(
        (seq.length - 1) * MODEL_SPEC.dModel,
        seq.length * MODEL_SPEC.dModel
      );
      sess.lastActivationVec = lastSlice;

      if (pacingModeRef.current !== 'turbo-0ms') {
        setNodeAComputeMs(nodeAMs);
        setLastActivation(lastSlice);
        setActiveStage('datachannel-tx');
      }

      const chunks = encodeActivationFullChunks(
        sess.stepId,
        seq.length,
        sess.temperature,
        fullAct,
        currentByte
      );
      sess.lastTxBytes = fullAct.byteLength;
      for (const ch of chunks) {
        mesh.sendFromNodeA(ch);
      }
      if (pacingModeRef.current !== 'turbo-0ms') {
        setChannelStats({ ...mesh.stats });
      }
    } else {
      let stepAct: Float32Array;
      let nodeAMs: number;
      const spec = specCacheRef.current;

      if (
        speculativeOverlapRef.current &&
        spec &&
        spec.pos === pos &&
        spec.predictedByte === currentByte
      ) {
        // Speculative Pipeline Overlap Hit! Layer 0-2 activation for `pos` was already pre-computed while Phone 2 was busy!
        stepAct = spec.activation;
        nodeAMs = 0.04;
        specCacheRef.current = null;
        setSpeculativeHits((h) => h + 1);
      } else {
        const t0 = performance.now();
        stepAct = nodeARuntimeRef.current.forwardStep(currentByte, pos);
        nodeAMs = performance.now() - t0;
        specCacheRef.current = null;
      }

      sess.lastNodeAMs = nodeAMs;
      sess.lastActivationVec = stepAct;
      if (pacingModeRef.current !== 'turbo-0ms') {
        setNodeAComputeMs(nodeAMs);
        setLastActivation(stepAct);
        setActiveStage('datachannel-tx');
      }

      const pkt =
        sess.transferMode === 'kv-int8'
          ? encodeActivationStepInt8Packet(
              sess.stepId,
              pos,
              sess.temperature,
              stepAct,
              currentByte
            )
          : encodeActivationStepPacket(
              sess.stepId,
              pos,
              sess.temperature,
              stepAct,
              currentByte
            );

      sess.lastTxBytes = sess.transferMode === 'kv-int8' ? 388 : stepAct.byteLength;
      mesh.sendFromNodeA(pkt);
      if (pacingModeRef.current !== 'turbo-0ms') {
        setChannelStats({ ...mesh.stats });
      }

      // Speculative Pipeline Overlap: While Phone 2 computes Layers 3-5 over the network,
      // Phone 1 speculatively pre-computes Layer 0-2 for the most likely next byte at `pos + 1`!
      if (
        speculativeOverlapRef.current &&
        typeof hintCandidateByte === 'number' &&
        pos + 1 < MODEL_SPEC.maxSeqLen &&
        sess.generatedCount + 1 < sess.targetNewBytes
      ) {
        const specAct = nodeARuntimeRef.current.forwardStep(hintCandidateByte, pos + 1);
        specCacheRef.current = {
          pos: pos + 1,
          predictedByte: hintCandidateByte,
          activation: specAct,
        };
      }
    }
  };

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
        await mesh.startBroadcastAutoPair(role, quickRoomCode);
      }
    }

    // If user clicked "Run Split Inference" while on Phone 2 (Node B), send start trigger to Phone 1!
    if (role === 'node-b') {
      setGeneratedHistory([]);
      setIsGenerating(true);
      setActiveStage('datachannel-rx');
      mesh.sendFromNodeB(encodeRemoteStartRequestPacket());
      return;
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
    setTtftMs(0);
    setSpeculativeHits(0);
    specCacheRef.current = null;

    nodeARuntimeRef.current.resetCache();
    nodeBRuntimeRef.current.resetCache();
    samplingPrngRef.current.manualSeed(MODEL_SPEC.seed);

    mesh.sendFromNodeA(
      encodeResetSessionPacket(Date.now() & 0xffffffff, initialBytes, maxNewBytes)
    );

    // Coalesced 1-Packet Prompt Prefill: send all 0..L-2 prompt activations in a single binary packet!
    if (transferMode !== 'full-sequence' && initialBytes.length > 1) {
      const prefillCount = initialBytes.length - 1;
      const prefillTensor = new Float32Array(prefillCount * MODEL_SPEC.dModel);
      for (let i = 0; i < prefillCount; i++) {
        const act = nodeARuntimeRef.current.forwardStep(initialBytes[i], i);
        prefillTensor.set(act, i * MODEL_SPEC.dModel);
      }
      mesh.sendFromNodeA(encodePrefillBatchPacket(prefillCount, prefillTensor));
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
      lastInputByte: initialBytes[initialBytes.length - 1],
      lastTxBytes: transferMode === 'kv-int8' ? 388 : 1536,
      temperature,
      transferMode,
    };

    setIsGenerating(true);
    executeNextNodeAStep();
  };

  startTriggerRef.current = () => {
    handleStartSplitInference();
  };

  const handleSelectRole = async (newRole: PeerRole) => {
    genSessionRef.current.active = false;
    setIsGenerating(false);
    setActiveStage('idle');
    setRole(newRole);
    if (!meshRef.current) return;
    if (newRole === 'dual-loopback') {
      await meshRef.current.connectDualNodeLoopback();
    } else {
      await meshRef.current.startBroadcastAutoPair(newRole, quickRoomCode);
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

  const handleChangeInitScheme = (scheme: WeightInitScheme) => {
    setInitScheme(scheme);
    setShardA(createShardAWeights(MODEL_SPEC.seed, undefined, scheme));
    setShardB(createShardBWeights(MODEL_SPEC.seed, undefined, scheme));
    setWeightsSourceLabel(
      scheme === 'pytorch-nn-default'
        ? 'PyTorch nn.Module Default (Kaiming Uniform + LN 1/0 · Seed 42)'
        : 'PyTorch torch.manual_seed(42) (True Box-Muller FP32)'
    );
  };

  const handleResetSeed42Weights = () => {
    setShardA(createShardAWeights(MODEL_SPEC.seed, undefined, initScheme));
    setShardB(createShardBWeights(MODEL_SPEC.seed, undefined, initScheme));
    setWeightsSourceLabel(
      initScheme === 'pytorch-nn-default'
        ? 'PyTorch nn.Module Default (Kaiming Uniform + LN 1/0 · Seed 42)'
        : 'PyTorch torch.manual_seed(42) (True Box-Muller FP32)'
    );
  };

  const allSequenceBytes = [
    ...promptByteList,
    ...generatedHistory.map((item) => item.byte),
  ];

  const promptTextDecoded = promptByteList.map((b) => formatByteGlyph(b)).join('');
  const generatedTextDecoded = generatedHistory.map((item) => formatByteGlyph(item.byte)).join('');

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
            data-testid="btn-run-inference"
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

      {/* Quick 1-Tap Device Role & Room Bar (Always visible so Phone 1 and Phone 2 pair in 1 tap) */}
      <div className="px-4 md:px-6 py-2.5 bg-[#111827] border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-slate-400">Select This Phone&apos;s Role:</span>
          <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-950 rounded-lg border border-slate-800">
            <button
              type="button"
              data-testid="btn-role-node-a"
              onClick={() => handleSelectRole('node-a')}
              className={`px-2.5 py-1 rounded font-mono text-xs transition-colors whitespace-nowrap shrink-0 ${
                role === 'node-a'
                  ? 'bg-cyan-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-slate-100'
              }`}
            >
              Phone 1 (Layers 0–2)
            </button>
            <button
              type="button"
              data-testid="btn-role-node-b"
              onClick={() => handleSelectRole('node-b')}
              className={`px-2.5 py-1 rounded font-mono text-xs transition-colors whitespace-nowrap shrink-0 ${
                role === 'node-b'
                  ? 'bg-amber-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-slate-100'
              }`}
            >
              Phone 2 (Layers 3–5)
            </button>
            <button
              type="button"
              data-testid="btn-role-loopback"
              onClick={() => handleSelectRole('dual-loopback')}
              className={`px-2.5 py-1 rounded font-mono text-xs transition-colors whitespace-nowrap shrink-0 ${
                role === 'dual-loopback'
                  ? 'bg-slate-700 text-slate-100 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Single-Phone Loopback
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 font-mono tabular-nums">
          {role !== 'dual-loopback' && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Room:</span>
              <input
                type="text"
                aria-label="Quick Pairing Room Code"
                value={quickRoomCode}
                onChange={(e) => setQuickRoomCode(e.target.value)}
                className="w-16 px-2 py-0.5 rounded bg-slate-950 border border-slate-700 text-cyan-300 text-xs"
              />
              <button
                type="button"
                onClick={() => meshRef.current?.startBroadcastAutoPair(role, quickRoomCode)}
                className="px-2 py-0.5 rounded border border-cyan-500/40 bg-cyan-950/40 text-cyan-200 hover:bg-cyan-900/50 whitespace-nowrap shrink-0"
              >
                Sync Room
              </button>
            </div>
          )}
          <span className="text-slate-400">
            Active RAM:{' '}
            <strong className="text-slate-200">
              {role === 'node-a'
                ? 'Shard A Only (22.47 MB)'
                : role === 'node-b'
                ? 'Shard B Only (21.69 MB)'
                : 'Loopback (44.16 MB)'}
            </strong>
          </span>
          <span>
            Status:{' '}
            <strong
              className={
                connectionState === 'connected'
                  ? 'text-emerald-400'
                  : connectionState === 'awaiting-peer'
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }
            >
              {connectionState === 'connected'
                ? role === 'dual-loopback'
                  ? 'LOCAL LOOPBACK READY'
                  : `P2P CONNECTED (Room #${quickRoomCode})`
                : connectionState.toUpperCase()}
            </strong>
          </span>
        </div>
      </div>

      {/* Main Workspace Container */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 md:px-6 py-5 space-y-6">
        {activeTab === 'workspace' && (
          <>
            {/* PROMINENT LIVE TOKEN PING-PONG & FINAL OUTPUT MONITOR (Visible at the top of BOTH Phone 1 and Phone 2) */}
            <section
              aria-label="Live P2P Token Activity and Generated Output"
              className="bg-[#111827] border border-slate-800 rounded-lg p-5 space-y-4"
            >
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div>
                  <h2 className="text-base font-semibold text-slate-100">
                    Live P2P Token Exchange &amp; Final Byte Output ({generatedHistory.length} / {maxNewBytes} Bytes Generated)
                  </h2>
                  <p className="text-xs text-slate-400 font-mono tabular-nums mt-0.5">
                    {isGenerating
                      ? `ACTIVE PASS: Step #${generatedHistory.length + 1} of ${maxNewBytes} · Stage: ${activeStage.toUpperCase()} · Speed: ${throughputBps.toFixed(1)} B/s`
                      : generatedHistory.length > 0
                      ? `COMPLETED: ${generatedHistory.length} bytes at ${throughputBps.toFixed(1)} B/s · TTFT: ${ttftMs.toFixed(1)} ms · Speculative Hits: ${speculativeHits}`
                      : 'READY: Tap "Execute P2P Split Pass" on either phone to stream activations & byte tokens'}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-3 text-xs font-mono tabular-nums">
                  <span className="text-emerald-400 font-semibold">
                    {throughputBps.toFixed(1)} tok/s
                  </span>
                  <span aria-hidden="true" className="text-slate-600">·</span>
                  <span className="text-slate-300">
                    TTFT: {ttftMs > 0 ? `${ttftMs.toFixed(1)} ms` : '—'}
                  </span>
                  <span aria-hidden="true" className="text-slate-600">·</span>
                  <span className="text-cyan-300">
                    TX: {channelStats.messagesSent} ({(channelStats.bytesSent / 1024).toFixed(1)} KB)
                  </span>
                  <span aria-hidden="true" className="text-slate-600">·</span>
                  <span className="text-amber-300">
                    RX: {channelStats.messagesReceived} ({(channelStats.bytesReceived / 1024).toFixed(1)} KB)
                  </span>
                  {generatedHistory.length > 0 && (
                    <button
                      type="button"
                      onClick={() => {
                        const report = {
                          timestamp: new Date().toISOString(),
                          architecture: MODEL_SPEC,
                          seed: 42,
                          initScheme,
                          computeBackend,
                          transferMode,
                          pacingMode,
                          speculativeOverlap,
                          shardAChecksum,
                          shardBChecksum,
                          promptText: promptTextDecoded,
                          generatedText: generatedTextDecoded,
                          generatedBytesHex: generatedHistory.map((h) => `0x${h.byte.toString(16).toUpperCase().padStart(2, '0')}`),
                          metrics: {
                            bytesGenerated: generatedHistory.length,
                            throughputBytesPerSec: Number(throughputBps.toFixed(2)),
                            ttftMs: Number(ttftMs.toFixed(2)),
                            nodeALastMs: Number(nodeAComputeMs.toFixed(2)),
                            nodeBLastMs: Number(nodeBComputeMs.toFixed(2)),
                            speculativeHits,
                            rtcMessagesSent: channelStats.messagesSent,
                            rtcBytesSent: channelStats.bytesSent,
                          },
                        };
                        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `splitbyte-benchmark-seed42.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      }}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 transition-colors whitespace-nowrap shrink-0"
                    >
                      Export Benchmark (.json)
                    </button>
                  )}
                </div>
              </div>

              {/* 3-Column Live Packet & Output Readout */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Box 1: Phone 1 -> Phone 2 (Current Input Byte & Layer 2 Activation) */}
                <div
                  className={`p-3.5 rounded-lg border font-mono ${
                    activeStage === 'node-a' || activeStage === 'datachannel-tx'
                      ? 'border-cyan-400 bg-cyan-950/25'
                      : 'border-slate-800 bg-slate-950/70'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs text-cyan-400 mb-1">
                    <span>Phone 1 (Layers 0–2) ➔ Phone 2</span>
                    <span className="tabular-nums">{nodeAComputeMs.toFixed(2)} ms</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-100 tabular-nums">
                    Input Token:{' '}
                    {lastInputByte !== null
                      ? `'${formatByteGlyph(lastInputByte)}' (0x${lastInputByte.toString(16).toUpperCase().padStart(2, '0')} · dec ${lastInputByte})`
                      : '—'}
                  </div>
                  <div className="text-xs text-slate-400 mt-1 tabular-nums">
                    {transferMode === 'kv-int8'
                      ? `Sent Int8[384]+Scale (388 B) at Pos #${Math.max(0, allSequenceBytes.length - 1)}`
                      : `Sent Float32Array[384] (1,536 B) at Pos #${Math.max(0, allSequenceBytes.length - 1)}`}
                  </div>
                </div>

                {/* Box 2: Phone 2 -> Phone 1 (Computed Layers 3-5 & Returned Next Byte Token) */}
                <div
                  className={`p-3.5 rounded-lg border font-mono ${
                    activeStage === 'node-b' || activeStage === 'datachannel-rx'
                      ? 'border-amber-400 bg-amber-950/25'
                      : 'border-slate-800 bg-slate-950/70'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs text-amber-400 mb-1">
                    <span>Phone 2 (Layers 3–5) ➔ Phone 1</span>
                    <span className="tabular-nums">{nodeBComputeMs.toFixed(2)} ms</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-100 tabular-nums">
                    Returned Token:{' '}
                    {lastOutputByte !== null
                      ? `'${formatByteGlyph(lastOutputByte)}' (0x${lastOutputByte.toString(16).toUpperCase().padStart(2, '0')} · dec ${lastOutputByte})`
                      : 'Awaiting Step...'}
                  </div>
                  <div className="text-xs text-slate-400 mt-1 tabular-nums">
                    Argmax over 256 Byte Logits · Seed 42 · Spec Hits: {speculativeHits}
                  </div>
                </div>

                {/* Box 3: Final / Live Combined Output String */}
                <div className="p-3.5 rounded-lg border border-slate-700 bg-slate-950 font-mono flex flex-col justify-between">
                  <div className="flex items-center justify-between text-xs text-emerald-400 mb-1">
                    <span>Live Decoded Output Sequence</span>
                    <span className="tabular-nums">{allSequenceBytes.length} Bytes Total</span>
                  </div>
                  <div
                    className="text-sm font-semibold text-slate-100 break-all"
                    data-testid="generated-byte-stream"
                  >
                    <span className="text-slate-400">{promptTextDecoded}</span>
                    <span className="text-cyan-300 underline decoration-cyan-500/50 underline-offset-4">
                      {generatedTextDecoded}
                    </span>
                    {generatedHistory.length === 0 && !isGenerating && (
                      <span className="text-slate-500 text-xs font-normal ml-1">
                        [Tap &quot;Execute P2P Split Pass&quot;]
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1 tabular-nums truncate">
                    Prompt: &quot;{promptTextDecoded}&quot; + {generatedHistory.length} generated bytes
                  </div>
                </div>
              </div>

              {/* Token-by-Token Visual Stream Tape (Shows every token received/generated in sequence) */}
              {generatedHistory.length > 0 && (
                <div className="pt-2">
                  <div className="text-xs font-mono text-slate-400 mb-2">
                    Step-by-Step Generated Byte Tape (Phone 1 L0–L2 ➔ Phone 2 L3–L5):
                  </div>
                  <div className="flex items-center gap-2 overflow-x-auto pb-2 font-mono text-xs tabular-nums">
                    {generatedHistory.map((item, idx) => (
                      <div
                        key={`${item.step}-${idx}`}
                        className="px-2.5 py-1.5 rounded border border-slate-800 bg-slate-950 shrink-0"
                      >
                        <div className="text-[10px] text-slate-400">
                          Step #{item.step} (In: &apos;{formatByteGlyph(item.inputByte)}&apos;)
                        </div>
                        <div className="text-sm font-bold text-cyan-300 mt-0.5">
                          &apos;{formatByteGlyph(item.byte)}&apos;{' '}
                          <span className="text-xs font-normal text-amber-300">
                            0x{item.byte.toString(16).toUpperCase().padStart(2, '0')}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* LEFT CONTROL & PARAMETER COLUMN (4 cols on 12-col grid) */}
              <aside className="lg:col-span-4 bg-[#111827] border border-slate-800 rounded-lg p-5 space-y-5">
                <div className="border-b border-slate-800 pb-3">
                  <h1 className="text-base font-semibold text-slate-100">
                    Split Inference &amp; Turbo Optimizations
                  </h1>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Configure byte prompt, temperature, Int8 activation compression, and speculative pipeline overlap
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
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
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
                  <label className="block text-xs font-medium text-slate-200 mb-1.5">
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

                {/* Execution Pacing & Speculative Pipeline Overlap */}
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-200 mb-1">
                      Execution Pacing
                    </label>
                    <div className="grid grid-cols-2 gap-1 p-1 bg-slate-950 rounded border border-slate-800">
                      <button
                        type="button"
                        onClick={() => setPacingMode('turbo-0ms')}
                        className={`py-1 text-[11px] font-mono rounded transition-colors ${
                          pacingMode === 'turbo-0ms'
                            ? 'bg-emerald-500 text-slate-950 font-semibold'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Turbo 0ms
                      </button>
                      <button
                        type="button"
                        onClick={() => setPacingMode('visual-60ms')}
                        className={`py-1 text-[11px] font-mono rounded transition-colors ${
                          pacingMode === 'visual-60ms'
                            ? 'bg-cyan-500 text-slate-950 font-semibold'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Demo 60ms
                      </button>
                    </div>
                  </div>

                  <div>
                    <label className="block text-[11px] font-medium text-slate-200 mb-1">
                      Speculative Overlap
                    </label>
                    <button
                      type="button"
                      onClick={() => setSpeculativeOverlap((prev) => !prev)}
                      className={`w-full py-1.5 px-2 rounded border text-[11px] font-mono transition-colors ${
                        speculativeOverlap
                          ? 'bg-cyan-950/60 border-cyan-500/60 text-cyan-300 font-semibold'
                          : 'bg-slate-950 border-slate-800 text-slate-400'
                      }`}
                    >
                      {speculativeOverlap ? '● Top-1 Lookahead ON' : '○ Lookahead OFF'}
                    </button>
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
                      setTransferMode(e.target.value as 'kv-step' | 'kv-int8' | 'full-sequence' | 'raw-1byte')
                    }
                    className="w-full px-3 py-2 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:border-cyan-400 focus:outline-none"
                  >
                    <option value="kv-step">KV-Cached FP32 [1 × 384 Float32] (1,536 B/step — Exact)</option>
                    <option value="kv-int8">KV-Cached Int8 Quantized [384 Int8 + Scale] (388 B/step — 75% Smaller)</option>
                    <option value="full-sequence">Full Sequence Tensor [1 × T × 384 Float32] (SCTP Chunked)</option>
                    <option value="raw-1byte">Strict 1-Byte Uint8Array([nextByte]) Reply Mode</option>
                  </select>
                </div>

                {/* Web CPU Compute Engine Selector (WASM vs JS) */}
                <div>
                  <label htmlFor="compute-backend-select" className="block text-xs font-medium text-slate-200 mb-1.5">
                    Web CPU Compute Backend (Zero WebGPU/WebGL)
                  </label>
                  <select
                    id="compute-backend-select"
                    value={computeBackend}
                    onChange={(e) => setComputeBackend(e.target.value as ComputeBackend)}
                    className="w-full px-3 py-2 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 focus:border-cyan-400 focus:outline-none"
                  >
                    <option value="wasm-cpu">WebAssembly + Fused QKV (8x Unrolled Linear Memory)</option>
                    <option value="js-cpu">JavaScript + Fused QKV (8x Unrolled Float32Array)</option>
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
                    <div className="overflow-x-auto max-h-64">
                      <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                        <thead>
                          <tr className="border-b border-slate-800 text-slate-400">
                            <th className="py-2 px-2.5">Step</th>
                            <th className="py-2 px-2.5">Pos</th>
                            <th className="py-2 px-2.5">Phone 1 Input Byte</th>
                            <th className="py-2 px-2.5">Phone 2 Output Byte</th>
                            <th className="py-2 px-2.5 text-right">Node A (L0–L2)</th>
                            <th className="py-2 px-2.5 text-right">DataChannel TX</th>
                            <th className="py-2 px-2.5 text-right">Node B (L3–L5)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/70 text-slate-200">
                          {generatedHistory.map((item, idx) => (
                            <tr key={`${item.step}-${idx}`} className="hover:bg-slate-900/50">
                              <td className="py-1.5 px-2.5 text-slate-400">#{item.step}</td>
                              <td className="py-1.5 px-2.5">{item.seqPos}</td>
                              <td className="py-1.5 px-2.5 text-slate-300">
                                &apos;{formatByteGlyph(item.inputByte)}&apos; (0x{item.inputByte.toString(16).toUpperCase().padStart(2, '0')})
                              </td>
                              <td className="py-1.5 px-2.5 text-cyan-300 font-semibold">
                                &apos;{formatByteGlyph(item.byte)}&apos; (0x{item.byte.toString(16).toUpperCase().padStart(2, '0')} · dec {item.byte})
                              </td>
                              <td className="py-1.5 px-2.5 text-right">
                                {item.nodeAMs > 0 ? `${item.nodeAMs.toFixed(2)} ms` : 'Remote'}
                              </td>
                              <td className="py-1.5 px-2.5 text-right text-slate-400">
                                {item.txBytes.toLocaleString()} B
                              </td>
                              <td className="py-1.5 px-2.5 text-right text-amber-300">
                                {item.nodeBMs.toFixed(2)} ms
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
          </>
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
              setQuickRoomCode(code);
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
            role={role}
            shardA={shardA}
            shardB={shardB}
            initScheme={initScheme}
            weightsSourceLabel={weightsSourceLabel}
            onChangeInitScheme={handleChangeInitScheme}
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
