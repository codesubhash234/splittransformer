import React, { useState } from 'react';
import { MODEL_SPEC, ShardAWeights, ShardBWeights } from '../engine/transformer';
import { PYTORCH_EXPORT_SCRIPT } from '../engine/pytorchScript';

interface ArchitectureSpecPanelProps {
  shardA: ShardAWeights;
  shardB: ShardBWeights;
  weightsSourceLabel: string;
  onImportWeightsBuffer: (buffer: ArrayBuffer, fileName: string) => void;
  onResetSeed42Weights: () => void;
}

export const ArchitectureSpecPanel: React.FC<ArchitectureSpecPanelProps> = ({
  shardA,
  shardB,
  weightsSourceLabel,
  onImportWeightsBuffer,
  onResetSeed42Weights,
}) => {
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadStatus(`Reading ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)...`);
    try {
      const buf = await file.arrayBuffer();
      if (
        buf.byteLength !== MODEL_SPEC.totalBytesFp32 &&
        buf.byteLength !== MODEL_SPEC.shardABytesFp32 &&
        buf.byteLength !== MODEL_SPEC.shardBBytesFp32
      ) {
        setUploadStatus(
          `Warning: ${file.name} is ${buf.byteLength.toLocaleString()} bytes. Expected 44,163,072 B (Full weights.bin), 22,473,216 B (Shard A), or 21,689,856 B (Shard B). Mapping available floats.`
        );
      } else {
        setUploadStatus(`Loaded ${file.name} (${buf.byteLength.toLocaleString()} bytes FP32) successfully.`);
      }
      onImportWeightsBuffer(buf, file.name);
    } catch (err) {
      setUploadStatus(err instanceof Error ? err.message : 'Failed to parse binary weights file.');
    }
  };

  const handleDownloadPyTorchScript = () => {
    const blob = new Blob([PYTORCH_EXPORT_SCRIPT], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'export_weights.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadShardBin = (which: 'shard-a' | 'shard-b') => {
    const arr = which === 'shard-a' ? shardA.rawBuffer : shardB.rawBuffer;
    const blob = new Blob([arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength)], {
      type: 'application/octet-stream',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = which === 'shard-a' ? 'weights_shard_a.bin' : 'weights_shard_b.bin';
    a.click();
    URL.revokeObjectURL(url);
  };

  const tensorRows = [
    {
      node: 'Phone 1 (Node A)',
      module: 'tok_emb.weight',
      shape: '[256, 384]',
      params: 98_304,
      bytes: 393_216,
    },
    {
      node: 'Phone 1 (Node A)',
      module: 'pos_emb.weight',
      shape: '[512, 384]',
      params: 196_608,
      bytes: 786_432,
    },
    {
      node: 'Phone 1 (Node A)',
      module: 'blocks[0..2] (3 × TransformerBlock)',
      shape: '3 × [Attn(591,360) + FFN(1,181,568) + LN(1,536)]',
      params: 5_323_392,
      bytes: 21_293_568,
    },
    {
      node: 'Phone 2 (Node B)',
      module: 'blocks[3..5] (3 × TransformerBlock)',
      shape: '3 × [Attn(591,360) + FFN(1,181,568) + LN(1,536)]',
      params: 5_323_392,
      bytes: 21_293_568,
    },
    {
      node: 'Phone 2 (Node B)',
      module: 'ln_f.weight + ln_f.bias',
      shape: '[384] + [384]',
      params: 768,
      bytes: 3_072,
    },
    {
      node: 'Phone 2 (Node B)',
      module: 'lm_head.weight (bias=False)',
      shape: '[256, 384]',
      params: 98_304,
      bytes: 393_216,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Parameter & Memory Allocation Summary */}
      <div className="bg-[#111827] border border-slate-800 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-slate-800">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Target Architecture &amp; Exact Parameter Specification (11,040,768 Parameters)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5 font-mono tabular-nums">
              Active Weights Source: {weightsSourceLabel} · Shard A CRC: {shardA.checksum} · Shard B CRC: {shardB.checksum}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onResetSeed42Weights}
              className="px-3 py-1.5 rounded border border-slate-700 bg-slate-900 hover:bg-slate-800 text-xs font-mono text-slate-200 transition-colors whitespace-nowrap shrink-0"
            >
              Restore Seed 42 Weights
            </button>
            <button
              type="button"
              onClick={() => handleDownloadShardBin('shard-a')}
              className="px-3 py-1.5 rounded border border-cyan-500/40 bg-cyan-950/30 hover:bg-cyan-900/40 text-xs font-mono text-cyan-200 transition-colors whitespace-nowrap shrink-0"
            >
              Export Shard A (.bin)
            </button>
            <button
              type="button"
              onClick={() => handleDownloadShardBin('shard-b')}
              className="px-3 py-1.5 rounded border border-amber-500/40 bg-amber-950/30 hover:bg-amber-900/40 text-xs font-mono text-amber-200 transition-colors whitespace-nowrap shrink-0"
            >
              Export Shard B (.bin)
            </button>
          </div>
        </div>

        {/* High-Density Parameter Breakdown Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 font-medium">Assigned Device Shard</th>
                <th className="py-2.5 px-3 font-medium">Module / Tensor</th>
                <th className="py-2.5 px-3 font-medium">Dimensions</th>
                <th className="py-2.5 px-3 font-medium text-right">FP32 Parameters</th>
                <th className="py-2.5 px-3 font-medium text-right">FP32 Bytes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 text-slate-200">
              {tensorRows.map((row) => (
                <tr key={row.module} className="hover:bg-slate-900/50">
                  <td
                    className={`py-2.5 px-3 ${
                      row.node.includes('Phone 1') ? 'text-cyan-400' : 'text-amber-400'
                    }`}
                  >
                    {row.node}
                  </td>
                  <td className="py-2.5 px-3 font-medium">{row.module}</td>
                  <td className="py-2.5 px-3 text-slate-400">{row.shape}</td>
                  <td className="py-2.5 px-3 text-right">{row.params.toLocaleString()}</td>
                  <td className="py-2.5 px-3 text-right">{row.bytes.toLocaleString()} B</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-slate-700 bg-slate-950/70 text-slate-100 font-semibold">
                <td className="py-3 px-3" colSpan={3}>
                  Total Distributed Model (Shard A: 5,618,304 + Shard B: 5,422,464)
                </td>
                <td className="py-3 px-3 text-right text-cyan-300">
                  {MODEL_SPEC.totalParams.toLocaleString()}
                </td>
                <td className="py-3 px-3 text-right text-amber-300">
                  {MODEL_SPEC.totalBytesFp32.toLocaleString()} B (44.16 MB)
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* Binary weights.bin Importer & Reference PyTorch Script */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Load Custom PyTorch weights.bin (FP32)
            </h3>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Upload a raw FP32 <code className="text-slate-200">weights.bin</code> (44,163,072 bytes) exported from PyTorch. SplitByte automatically slices bytes <code className="text-cyan-300">0..22,473,215</code> for Phone 1 and bytes <code className="text-amber-300">22,473,216..44,163,071</code> for Phone 2.
            </p>

            <label
              htmlFor="weights-bin-input"
              className="mt-4 flex flex-col items-center justify-center p-6 border border-dashed border-slate-700 hover:border-cyan-400 rounded-lg bg-slate-950/60 cursor-pointer transition-colors"
            >
              <span className="text-xs font-medium text-slate-200">
                Select or Drop weights.bin File
              </span>
              <span className="text-[11px] font-mono text-slate-400 mt-1 tabular-nums">
                11,040,768 floats · 44.16 MB FP32
              </span>
              <input
                id="weights-bin-input"
                type="file"
                accept=".bin,application/octet-stream"
                onChange={handleFileChange}
                className="sr-only"
              />
            </label>

            {uploadStatus && (
              <div className="mt-3 p-2.5 rounded border border-slate-800 bg-slate-950 text-xs font-mono text-cyan-300">
                {uploadStatus}
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Zero WebGPU / Zero WebGL</span>
            <span>Strict Float32Array CPU</span>
          </div>
        </div>

        <div className="lg:col-span-7 bg-[#111827] border border-slate-800 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">
                  Reference PyTorch Seed 42 Weight Generator
                </h3>
                <p className="text-xs text-slate-400">
                  Exact specification script for deterministic <code className="text-slate-200">weights.bin</code> creation
                </p>
              </div>
              <button
                type="button"
                onClick={handleDownloadPyTorchScript}
                className="px-3 py-1.5 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-colors whitespace-nowrap shrink-0"
              >
                Download export_weights.py
              </button>
            </div>

            <pre className="p-3 rounded bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-56 leading-relaxed">
              {PYTORCH_EXPORT_SCRIPT}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
