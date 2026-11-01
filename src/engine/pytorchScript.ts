/**
 * Reference PyTorch `torch.manual_seed(42)` Exporter Script & Deterministic Pairing Matrix Generator
 */

export const PYTORCH_EXPORT_SCRIPT = `#!/usr/bin/env python3
"""
Deterministic Weight Generator for 6-Layer Byte-Level Causal Micro-Transformer
Architecture Specification:
  - Vocab Size: 256 (Direct ASCII / UTF-8 bytes [0..255])
  - Context Length (max_seq_len): 512
  - Layers (n_layers): 6 (Phone 1 runs Layers 0-2, Phone 2 runs Layers 3-5)
  - Hidden Dimension (d_model): 384
  - Attention Heads (n_heads): 6 (Head dimension = 64)
  - Feed-Forward Dimension (d_ff): 1536 (4 * d_model)
  - Total Parameters: Exactly 11,040,768 FP32 parameters (44,163,072 bytes)
"""

import struct
import torch
import torch.nn as nn

torch.manual_seed(42)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=384, n_heads=6, d_ff=1536):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

class ByteMicroTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(256, 384)
        self.pos_emb = nn.Embedding(512, 384)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(6)])
        self.ln_f = nn.LayerNorm(384, eps=1e-5)
        self.lm_head = nn.Linear(384, 256, bias=False)

if __name__ == "__main__":
    model = ByteMicroTransformer()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 11_040_768, f"Expected 11,040,768 params, got {total_params}"
    print(f"Verified Total Parameters: {total_params:,} ({total_params * 4:,} bytes FP32)")

    with open("weights.bin", "wb") as f:
        for name, param in model.named_parameters():
            tensor_fp32 = param.detach().cpu().contiguous().to(torch.float32)
            f.write(tensor_fp32.numpy().tobytes())

    print("Exported weights.bin (44,163,072 bytes) successfully.")
`;

/**
 * Generates a deterministic 25x25 QR-style finder & payload matrix from a pairing token or URL
 * so the UI can render crisp, offline SVG pairing codes with zero external requests.
 */
export function generatePairingMatrix(payload: string, size = 25): boolean[][] {
  const grid: boolean[][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => false)
  );

  // Draw standard 7x7 finder patterns in 3 corners
  const drawFinder = (r0: number, c0: number) => {
    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        const isBorder = r === 0 || r === 6 || c === 0 || c === 6;
        const isInner = r >= 2 && r <= 4 && c >= 2 && c <= 4;
        grid[r0 + r][c0 + c] = isBorder || isInner;
      }
    }
  };

  drawFinder(0, 0);
  drawFinder(0, size - 7);
  drawFinder(size - 7, 0);

  // Timing patterns
  for (let i = 8; i < size - 8; i++) {
    grid[6][i] = i % 2 === 0;
    grid[i][6] = i % 2 === 0;
  }

  // Fill data cells deterministically from payload bytes
  let bitIdx = 0;
  const totalBits = payload.length * 8;
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const inTopLeft = r < 8 && c < 8;
      const inTopRight = r < 8 && c >= size - 8;
      const inBottomLeft = r >= size - 8 && c < 8;
      if (inTopLeft || inTopRight || inBottomLeft || r === 6 || c === 6) {
        continue;
      }
      if (totalBits === 0) {
        grid[r][c] = (r + c) % 2 === 0;
      } else {
        const charIndex = Math.floor(bitIdx / 8) % payload.length;
        const shift = bitIdx % 8;
        const code = payload.charCodeAt(charIndex);
        grid[r][c] = (((code >> shift) ^ (r * 3 + c * 7)) & 1) === 1;
        bitIdx++;
      }
    }
  }

  return grid;
}
