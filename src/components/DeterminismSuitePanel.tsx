import React, { useState } from 'react';
import {
  NodeARuntime,
  NodeBRuntime,
  ShardAWeights,
  ShardBWeights,
  sampleNextByte,
  formatByteGlyph,
} from '../engine/transformer';
import { getWasmCpuKernel } from '../engine/wasmKernel';

interface DeterminismSuitePanelProps {
  shardA: ShardAWeights;
  shardB: ShardBWeights;
}

interface BenchmarkStepRecord {
  step: number;
  pos: number;
  kvByte: number;
  fullSeqByte: number;
  wasmByte: number;
  kvLogitMax: number;
  fullSeqLogitMax: number;
  maxAbsDelta: number;
  nodeAMs: number;
  nodeBMs: number;
}

export const DeterminismSuitePanel: React.FC<DeterminismSuitePanelProps> = ({
  shardA,
  shardB,
}) => {
  const [testPrompt, setTestPrompt] = useState('SplitByte');
  const [numBytesToVerify, setNumBytesToVerify] = useState(8);
  const [isRunning, setIsRunning] = useState(false);
  const [records, setRecords] = useState<BenchmarkStepRecord[]>([]);
  const [summary, setSummary] = useState<{
    passed: boolean;
    totalMs: number;
    bytesPerSec: number;
    maxDelta: number;
    byteSequenceHex: string;
    wasmActive: boolean;
  } | null>(null);

  const runVerificationSuite = () => {
    setIsRunning(true);
    setTimeout(() => {
      const encoder = new TextEncoder();
      const promptBytes = Array.from(encoder.encode(testPrompt || 'A'));

      // Run 1: KV-Cached Split Execution with WebAssembly CPU Kernel
      const nodeAKvWasm = new NodeARuntime(shardA, 'wasm-cpu');
      const nodeBKvWasm = new NodeBRuntime(shardB, 'wasm-cpu');
      nodeAKvWasm.resetCache();
      nodeBKvWasm.resetCache();

      // Run 2: KV-Cached Split Execution with Pure JS 4x-Unrolled CPU Kernel
      const nodeAKvJs = new NodeARuntime(shardA, 'js-cpu');
      const nodeBKvJs = new NodeBRuntime(shardB, 'js-cpu');
      nodeAKvJs.resetCache();
      nodeBKvJs.resetCache();

      // Run 3: Full-Sequence [1, T, 384] Recomputed Split Execution
      const nodeAFull = new NodeARuntime(shardA, 'wasm-cpu');
      const nodeBFull = new NodeBRuntime(shardB, 'wasm-cpu');

      const seqKv = [...promptBytes];
      const seqFull = [...promptBytes];
      const stepRecords: BenchmarkStepRecord[] = [];

      const tStart = performance.now();

      // Prefill KV cache up to promptBytes.length - 1
      for (let i = 0; i < promptBytes.length - 1; i++) {
        const actW = nodeAKvWasm.forwardStep(seqKv[i], i);
        nodeBKvWasm.forwardStep(actW, i);
        const actJ = nodeAKvJs.forwardStep(seqKv[i], i);
        nodeBKvJs.forwardStep(actJ, i);
      }

      let worstDelta = 0;
      let allMatched = true;

      for (let step = 0; step < numBytesToVerify; step++) {
        const pos = seqKv.length - 1;
        const inputByte = seqKv[pos];

        const tA0 = performance.now();
        const actKvWasm = nodeAKvWasm.forwardStep(inputByte, pos);
        const tA1 = performance.now();

        const resKvWasm = nodeBKvWasm.forwardStep(actKvWasm, pos);
        const tB1 = performance.now();

        const sampledWasm = sampleNextByte(resKvWasm.logits, 0.0);

        // JS CPU pass
        const actKvJs = nodeAKvJs.forwardStep(inputByte, pos);
        const resKvJs = nodeBKvJs.forwardStep(actKvJs, pos);
        const sampledJs = sampleNextByte(resKvJs.logits, 0.0);

        // Full-sequence evaluation on `seqFull`
        const actFull = nodeAFull.forwardFullSequence(seqFull, false);
        const resFull = nodeBFull.forwardFullSequence(actFull, seqFull.length);
        const sampledFull = sampleNextByte(resFull.logits, 0.0);

        let maxAbsDelta = 0;
        for (let v = 0; v < 256; v++) {
          const d = Math.max(
            Math.abs(resKvWasm.logits[v] - resFull.logits[v]),
            Math.abs(resKvWasm.logits[v] - resKvJs.logits[v])
          );
          if (d > maxAbsDelta) maxAbsDelta = d;
        }
        if (maxAbsDelta > worstDelta) worstDelta = maxAbsDelta;
        if (
          sampledWasm.nextByte !== sampledFull.nextByte ||
          sampledWasm.nextByte !== sampledJs.nextByte
        ) {
          allMatched = false;
        }

        stepRecords.push({
          step: step + 1,
          pos,
          kvByte: sampledJs.nextByte,
          fullSeqByte: sampledFull.nextByte,
          wasmByte: sampledWasm.nextByte,
          kvLogitMax: sampledWasm.topK[0].logit,
          fullSeqLogitMax: sampledFull.topK[0].logit,
          maxAbsDelta,
          nodeAMs: tA1 - tA0,
          nodeBMs: tB1 - tA1,
        });

        seqKv.push(sampledWasm.nextByte);
        seqFull.push(sampledFull.nextByte);
      }

      const totalMs = performance.now() - tStart;
      const generatedBytes = stepRecords.map((r) => r.wasmByte);
      const byteSequenceHex = generatedBytes
        .map((b) => '0x' + b.toString(16).toUpperCase().padStart(2, '0'))
        .join(' ');

      const wasmInfo = getWasmCpuKernel();

      setRecords(stepRecords);
      setSummary({
        passed: allMatched && worstDelta < 1e-4,
        totalMs,
        bytesPerSec: (numBytesToVerify / Math.max(1, totalMs)) * 1000,
        maxDelta: worstDelta,
        byteSequenceHex,
        wasmActive: wasmInfo.isSupported,
      });
      setIsRunning(false);
    }, 20);
  };

  return (
    <div className="space-y-6">
      <div className="bg-[#111827] border border-slate-800 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-slate-800">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Seed 42 Determinism &amp; WebAssembly / Full-Sequence Equivalence Suite (Temperature = 0.0)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Verifies that WebAssembly (WASM CPU), 4x-Unrolled JS CPU, and Full-Sequence [1 × T × 384] Split Execution yield identical deterministic byte sequences
            </p>
          </div>
          <button
            type="button"
            onClick={runVerificationSuite}
            disabled={isRunning}
            className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-semibold text-xs transition-colors whitespace-nowrap shrink-0"
          >
            {isRunning ? 'Executing Verification...' : 'Run Determinism Benchmark'}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <div>
            <label htmlFor="det-prompt-input" className="block text-xs font-medium text-slate-300 mb-1.5">
              Verification Seed Prompt (UTF-8 Bytes)
            </label>
            <input
              id="det-prompt-input"
              type="text"
              value={testPrompt}
              onChange={(e) => setTestPrompt(e.target.value)}
              className="w-full px-3 py-2 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
            />
          </div>
          <div>
            <label htmlFor="det-steps-select" className="block text-xs font-medium text-slate-300 mb-1.5">
              Deterministic Bytes to Generate
            </label>
            <select
              id="det-steps-select"
              value={numBytesToVerify}
              onChange={(e) => setNumBytesToVerify(Number(e.target.value))}
              className="w-full px-3 py-2 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100"
            >
              <option value={4}>4 Bytes</option>
              <option value={8}>8 Bytes</option>
              <option value={16}>16 Bytes</option>
              <option value={24}>24 Bytes</option>
            </select>
          </div>
          <div className="flex flex-col justify-end">
            <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-xs font-mono tabular-nums flex items-center justify-between">
              <span className="text-slate-400">Sampling Mode:</span>
              <span className="text-emerald-400">Temperature = 0.0 (Greedy Argmax)</span>
            </div>
          </div>
        </div>

        {summary && (
          <div className="mb-5 p-4 rounded-lg border border-slate-800 bg-slate-950/90 space-y-2 font-mono text-xs tabular-nums">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className={summary.passed ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                {summary.passed
                  ? '● DETERMINISM VERIFIED: 100% Byte & Logit Equivalence (WASM CPU == JS CPU == Full-Sequence Split)'
                  : '✖ MISMATCH DETECTED'}
              </span>
              <span className="text-slate-300">
                Max Logit Delta: {summary.maxDelta.toExponential(3)} · Total Time: {summary.totalMs.toFixed(1)} ms
              </span>
            </div>
            <div className="text-slate-400">
              Deterministic Output Sequence: <span className="text-cyan-300">{summary.byteSequenceHex}</span>
            </div>
          </div>
        )}

        {records.length === 0 ? (
          <div className="py-10 text-center border border-dashed border-slate-800 rounded-lg text-xs font-mono text-slate-400">
            Click &quot;Run Determinism Benchmark&quot; to execute parallel WebAssembly CPU, JS CPU, and Full-Sequence split inference passes.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 px-3">Step</th>
                  <th className="py-2 px-3">Seq Pos</th>
                  <th className="py-2 px-3">WASM CPU Byte</th>
                  <th className="py-2 px-3">JS CPU Byte</th>
                  <th className="py-2 px-3">Full-Tensor [1×T×384] Byte</th>
                  <th className="py-2 px-3 text-right">Max Logit</th>
                  <th className="py-2 px-3 text-right">Logit Δ</th>
                  <th className="py-2 px-3 text-right">Phone 1 (ms)</th>
                  <th className="py-2 px-3 text-right">Phone 2 (ms)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 text-slate-200">
                {records.map((rec) => (
                  <tr key={rec.step} className="hover:bg-slate-900/50">
                    <td className="py-2 px-3 text-slate-400">#{rec.step}</td>
                    <td className="py-2 px-3">{rec.pos}</td>
                    <td className="py-2 px-3 text-emerald-400 font-semibold">
                      {rec.wasmByte} ({formatByteGlyph(rec.wasmByte)})
                    </td>
                    <td className="py-2 px-3 text-cyan-300">
                      {rec.kvByte} ({formatByteGlyph(rec.kvByte)})
                    </td>
                    <td className="py-2 px-3 text-amber-300">
                      {rec.fullSeqByte} ({formatByteGlyph(rec.fullSeqByte)})
                    </td>
                    <td className="py-2 px-3 text-right">{rec.kvLogitMax.toFixed(5)}</td>
                    <td className="py-2 px-3 text-right text-emerald-400">
                      {rec.maxAbsDelta.toExponential(2)}
                    </td>
                    <td className="py-2 px-3 text-right">{rec.nodeAMs.toFixed(2)} ms</td>
                    <td className="py-2 px-3 text-right">{rec.nodeBMs.toFixed(2)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
