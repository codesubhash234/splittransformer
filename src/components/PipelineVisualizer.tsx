import React from 'react';
import { MODEL_SPEC, formatByteGlyph } from '../engine/transformer';
import { PeerRole, ConnectionState, DataChannelStats } from '../engine/webrtc';

interface PipelineVisualizerProps {
  role: PeerRole;
  connectionState: ConnectionState;
  isGenerating: boolean;
  activeStage: 'idle' | 'node-a' | 'datachannel-tx' | 'node-b' | 'datachannel-rx';
  currentPos: number;
  lastInputByte: number | null;
  lastOutputByte: number | null;
  lastActivation: Float32Array | null;
  topK: Array<{ byte: number; logit: number; prob: number }>;
  nodeAComputeMs: number;
  nodeBComputeMs: number;
  stats: DataChannelStats;
  shardAChecksum: string;
  shardBChecksum: string;
  transferMode: 'kv-step' | 'kv-int8' | 'full-sequence' | 'raw-1byte';
}

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({
  role,
  connectionState,
  isGenerating,
  activeStage,
  currentPos,
  lastInputByte,
  lastOutputByte,
  lastActivation,
  topK,
  nodeAComputeMs,
  nodeBComputeMs,
  stats,
  shardAChecksum,
  shardBChecksum,
  transferMode,
}) => {
  // Compute min/max/norm of the 384-d activation vector crossing the WebRTC DataChannel
  let actMin = 0;
  let actMax = 0;
  let actL2 = 0;
  if (lastActivation && lastActivation.length >= 384) {
    actMin = Infinity;
    actMax = -Infinity;
    let sq = 0;
    for (let i = 0; i < 384; i++) {
      const v = lastActivation[i];
      if (v < actMin) actMin = v;
      if (v > actMax) actMax = v;
      sq += v * v;
    }
    actL2 = Math.sqrt(sq);
  }

  const nodeAActive = role === 'node-a' || role === 'dual-loopback';
  const nodeBActive = role === 'node-b' || role === 'dual-loopback';

  return (
    <div className="space-y-6">
      {/* Split Pipeline Topology Diagram */}
      <div className="bg-[#111827] border border-slate-800 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-slate-800">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Distributed Pipeline Topology (Layers 0–2 ⇄ WebRTC DataChannel ⇄ Layers 3–5)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5 font-mono tabular-nums">
              Total Parameters: {MODEL_SPEC.totalParams.toLocaleString()} FP32 · 44.16 MB · Seed: 42 · Context: {currentPos} / {MODEL_SPEC.maxSeqLen}
            </p>
          </div>
          <div className="text-xs font-mono tabular-nums text-slate-300">
            {connectionState === 'connected' ? (
              <span className="text-emerald-400">● RTCDataChannel Open ({stats.iceCandidateType})</span>
            ) : connectionState === 'connecting' || connectionState === 'gathering-ice' ? (
              <span className="text-amber-400">▲ Handshake In Progress</span>
            ) : (
              <span className="text-rose-400">✖ Peer Disconnected</span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-11 gap-4 items-stretch">
          {/* PHONE 1: NODE A (Layers 0-2) */}
          <div
            className={`lg:col-span-4 rounded-lg border p-4 transition-colors ${
              activeStage === 'node-a'
                ? 'border-cyan-400 bg-slate-900/90'
                : nodeAActive
                ? 'border-slate-700 bg-slate-900/60'
                : 'border-slate-800/70 bg-slate-950/40 opacity-60'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-xs font-mono text-cyan-400">
                  Phone 1 · Node A {nodeAActive ? '(Local CPU)' : '(Remote Peer)'}
                </div>
                <h3 className="text-sm font-semibold text-slate-100 mt-0.5">
                  Embeddings + Layers 0–2
                </h3>
              </div>
              <div className="text-right font-mono tabular-nums">
                <div className="text-xs text-slate-300">5,618,304 params</div>
                <div className="text-[11px] text-slate-400">22.47 MB FP32</div>
              </div>
            </div>

            <div className="space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between px-2.5 py-1.5 rounded bg-slate-950/80 border border-slate-800">
                <span className="text-slate-300">tok_emb [256×384] + pos_emb [512×384]</span>
                <span className="text-slate-400 tabular-nums">294,912</span>
              </div>
              {[0, 1, 2].map((layerIdx) => (
                <div
                  key={layerIdx}
                  className={`flex items-center justify-between px-2.5 py-1.5 rounded border transition-colors ${
                    activeStage === 'node-a' && isGenerating
                      ? 'bg-cyan-950/30 border-cyan-500/50 text-cyan-200'
                      : 'bg-slate-950/60 border-slate-800 text-slate-300'
                  }`}
                >
                  <span>Layer {layerIdx} · 6-Head Causal Attn + FFN</span>
                  <span className="text-slate-400 tabular-nums">1,774,464</span>
                </div>
              ))}
            </div>

            <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono tabular-nums text-slate-400">
              <span>Shard A CRC: {shardAChecksum}</span>
              <span className="text-cyan-300">{nodeAComputeMs.toFixed(2)} ms/step</span>
            </div>
          </div>

          {/* WEBRTC DATACHANNEL BRIDGE */}
          <div className="lg:col-span-3 flex flex-col justify-between rounded-lg border border-slate-800 bg-slate-950/80 p-4">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                <span>WebRTC DataChannel</span>
                <span className="text-cyan-400">SCTP / Binary</span>
              </div>

              {/* Forward Activation Arrow: Phone 1 -> Phone 2 */}
              <div
                className={`p-2.5 rounded border mb-2.5 transition-colors ${
                  activeStage === 'datachannel-tx' || activeStage === 'node-b'
                    ? 'border-cyan-500/60 bg-cyan-950/20'
                    : 'border-slate-800 bg-slate-900/50'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] font-mono">
                  <span className="text-slate-300">Layer 2 Activations →</span>
                  <span className="text-cyan-300 tabular-nums">
                    {transferMode === 'kv-int8'
                      ? '388 B (Int8)'
                      : transferMode === 'kv-step' || transferMode === 'raw-1byte'
                      ? '1,536 B'
                      : `${((currentPos || 1) * 384 * 4).toLocaleString()} B`}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-400 mt-1 tabular-nums">
                  {transferMode === 'kv-int8'
                    ? 'Int8Array [1 × 384] + FP32 scale'
                    : `Float32Array [${
                        transferMode === 'kv-step' || transferMode === 'raw-1byte'
                          ? '1 × 384'
                          : `1 × ${Math.max(1, currentPos)} × 384`
                      }]`}
                </div>
              </div>

              {/* Return Token Arrow: Phone 2 -> Phone 1 */}
              <div
                className={`p-2.5 rounded border transition-colors ${
                  activeStage === 'datachannel-rx'
                    ? 'border-amber-500/60 bg-amber-950/20'
                    : 'border-slate-800 bg-slate-900/50'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] font-mono">
                  <span className="text-slate-300">← Next Byte Token</span>
                  <span className="text-amber-300 tabular-nums">
                    {lastOutputByte !== null ? `0x${lastOutputByte.toString(16).toUpperCase().padStart(2, '0')} (${formatByteGlyph(lastOutputByte)})` : '—'}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-400 mt-1 tabular-nums">
                  Uint8Array [1 byte] · Input: {lastInputByte !== null ? `${lastInputByte} (${formatByteGlyph(lastInputByte)})` : '—'}
                </div>
              </div>
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono tabular-nums text-slate-400">
              <div>
                <div>TX Packets: {stats.messagesSent}</div>
                <div>RX Packets: {stats.messagesReceived}</div>
              </div>
              <div className="text-right">
                <div>TX: {(stats.bytesSent / 1024).toFixed(1)} KB</div>
                <div>RX: {(stats.bytesReceived / 1024).toFixed(1)} KB</div>
              </div>
            </div>
          </div>

          {/* PHONE 2: NODE B (Layers 3-5) */}
          <div
            className={`lg:col-span-4 rounded-lg border p-4 transition-colors ${
              activeStage === 'node-b'
                ? 'border-amber-400 bg-slate-900/90'
                : nodeBActive
                ? 'border-slate-700 bg-slate-900/60'
                : 'border-slate-800/70 bg-slate-950/40 opacity-60'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-xs font-mono text-amber-400">
                  Phone 2 · Node B {nodeBActive ? '(Local CPU)' : '(Remote Peer)'}
                </div>
                <h3 className="text-sm font-semibold text-slate-100 mt-0.5">
                  Layers 3–5 + LM Head
                </h3>
              </div>
              <div className="text-right font-mono tabular-nums">
                <div className="text-xs text-slate-300">5,422,464 params</div>
                <div className="text-[11px] text-slate-400">21.69 MB FP32</div>
              </div>
            </div>

            <div className="space-y-2 text-xs font-mono">
              {[3, 4, 5].map((layerIdx) => (
                <div
                  key={layerIdx}
                  className={`flex items-center justify-between px-2.5 py-1.5 rounded border transition-colors ${
                    activeStage === 'node-b' && isGenerating
                      ? 'bg-amber-950/30 border-amber-500/50 text-amber-200'
                      : 'bg-slate-950/60 border-slate-800 text-slate-300'
                  }`}
                >
                  <span>Layer {layerIdx} · 6-Head Causal Attn + FFN</span>
                  <span className="text-slate-400 tabular-nums">1,774,464</span>
                </div>
              ))}
              <div className="flex items-center justify-between px-2.5 py-1.5 rounded bg-slate-950/80 border border-slate-800">
                <span className="text-slate-300">ln_f [384] + lm_head [384×256]</span>
                <span className="text-slate-400 tabular-nums">99,072</span>
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono tabular-nums text-slate-400">
              <span>Shard B CRC: {shardBChecksum}</span>
              <span className="text-amber-300">{nodeBComputeMs.toFixed(2)} ms/step</span>
            </div>
          </div>
        </div>
      </div>

      {/* Live Intermediate Activation Tensor Heatmap (384 dimensions) & Top-5 Byte Logits */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* 384-D Layer 2 Activation Tensor Inspector */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">
                  Layer 2 Intermediate Activation Tensor (d_model = 384)
                </h3>
                <p className="text-xs text-slate-400">
                  Real-time FP32 activation vector serialized across WebRTC DataChannel (24 × 16 matrix)
                </p>
              </div>
              <div className="text-xs font-mono tabular-nums text-slate-300">
                L2 Norm: {actL2.toFixed(3)} · Range: [{actMin.toFixed(2)}, {actMax.toFixed(2)}]
              </div>
            </div>

            {/* 24 cols x 16 rows = 384 cells */}
            <div
              aria-hidden="true"
              className="grid gap-[2px] bg-slate-950 p-2 rounded border border-slate-800"
              style={{ gridTemplateColumns: 'repeat(24, minmax(0, 1fr))' }}
              data-testid="activation-heatmap"
            >
              {Array.from({ length: 384 }, (_, idx) => {
                const val = lastActivation ? lastActivation[idx] : 0;
                const clamped = Math.max(-2.5, Math.min(2.5, val));
                const intensity = Math.abs(clamped) / 2.5;
                const bg =
                  !lastActivation
                    ? 'rgba(30, 41, 59, 0.5)'
                    : clamped >= 0
                    ? `rgba(6, 182, 212, ${0.15 + intensity * 0.85})`
                    : `rgba(245, 158, 11, ${0.15 + intensity * 0.85})`;
                return (
                  <div
                    key={idx}
                    className="h-3 rounded-[1px]"
                    style={{ backgroundColor: bg }}
                  />
                );
              })}
            </div>
          </div>

          <div className="mt-3 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Negative Activations (Amber)</span>
            <span>0.0 (Obsidian)</span>
            <span>Positive Activations (Cyan)</span>
          </div>
        </div>

        {/* Phone 2 Output Logit Distribution (Top-5 Bytes out of 256 Vocab) */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">
                  Node B Output Byte Logits (Vocab = 256)
                </h3>
                <p className="text-xs text-slate-400">
                  Top-5 candidate bytes after Layer 5 + Final LayerNorm + LM Head
                </p>
              </div>
              <span className="text-xs font-mono text-amber-400 tabular-nums">
                Argmax [0..255]
              </span>
            </div>

            {topK.length === 0 ? (
              <div className="h-44 flex items-center justify-center border border-dashed border-slate-800 rounded text-xs font-mono text-slate-400">
                Run split inference to inspect live 256-byte logit probabilities.
              </div>
            ) : (
              <div className="space-y-2.5 mt-2">
                {topK.map((item, rank) => {
                  const pct = Math.max(2, Math.min(100, item.prob * 100));
                  return (
                    <div
                      key={item.byte}
                      className={`p-2.5 rounded border font-mono text-xs ${
                        rank === 0
                          ? 'border-amber-500/50 bg-amber-950/20 text-slate-100'
                          : 'border-slate-800 bg-slate-950/60 text-slate-300'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1 tabular-nums">
                        <span>
                          #{rank + 1} · Byte {item.byte} ({formatByteGlyph(item.byte)})
                        </span>
                        <span>
                          Logit: {item.logit >= 0 ? '+' : ''}
                          {item.logit.toFixed(4)} · {(item.prob * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-900 rounded overflow-hidden">
                        <div
                          className={rank === 0 ? 'h-full bg-amber-400' : 'h-full bg-cyan-500/70'}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="mt-3 pt-2.5 border-t border-slate-800 text-xs font-mono text-slate-400 tabular-nums flex items-center justify-between">
            <span>Selected Byte: {lastOutputByte !== null ? `${lastOutputByte} (${formatByteGlyph(lastOutputByte)})` : '—'}</span>
            <span>Deterministic Seed: 42</span>
          </div>
        </div>
      </div>
    </div>
  );
};
