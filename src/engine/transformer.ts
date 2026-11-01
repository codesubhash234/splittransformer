/**
 * Byte-Level Causal Decoder Micro-Transformer (GPT-style)
 * Exact Architecture Specification:
 * - Vocab Size: 256 (Direct ASCII / UTF-8 bytes [0..255])
 * - Context Length (max_seq_len): 512
 * - Layers (n_layers): 6 (Node A / Phone 1: Layers 0..2, Node B / Phone 2: Layers 3..5)
 * - Hidden Dimension (d_model): 384
 * - Attention Heads (n_heads): 6 (d_head = 64)
 * - Feed-Forward Dimension (d_ff): 1536 (4 * d_model)
 * - Total Parameters: 11,040,768 FP32 parameters (44,163,072 bytes = 44.16 MB)
 */

import { TorchMT19937, computeTensorChecksum } from './prng';

export const MODEL_SPEC = {
  vocabSize: 256,
  maxSeqLen: 512,
  nLayers: 6,
  nLayersPerNode: 3,
  dModel: 384,
  nHeads: 6,
  dHead: 64,
  dFf: 1536,
  layerNormEps: 1e-5,
  seed: 42,
  // Exact parameter counts:
  tokEmbParams: 256 * 384, // 98,304
  posEmbParams: 512 * 384, // 196,608
  paramsPerLayer: 1_774_464, // ln1(768) + attn(591,360) + ln2(768) + mlp(1,181,568)
  lnFParams: 768, // 384 + 384
  lmHeadParams: 256 * 384, // 98,304 (bias=False)
  shardAParams: 5_618_304, // tok_emb + pos_emb + 3 * paramsPerLayer
  shardBParams: 5_422_464, // 3 * paramsPerLayer + ln_f + lm_head
  totalParams: 11_040_768, // shardAParams + shardBParams
  totalBytesFp32: 44_163_072, // 11,040,768 * 4
  shardABytesFp32: 22_473_216, // 5,618,304 * 4
  shardBBytesFp32: 21_689_856, // 5,422,464 * 4
} as const;

export interface TransformerLayerWeights {
  layerIndex: number;
  ln1Weight: Float32Array; // [384]
  ln1Bias: Float32Array; // [384]
  qWeight: Float32Array; // [384, 384]
  qBias: Float32Array; // [384]
  kWeight: Float32Array; // [384, 384]
  kBias: Float32Array; // [384]
  vWeight: Float32Array; // [384, 384]
  vBias: Float32Array; // [384]
  outWeight: Float32Array; // [384, 384]
  outBias: Float32Array; // [384]
  ln2Weight: Float32Array; // [384]
  ln2Bias: Float32Array; // [384]
  fc1Weight: Float32Array; // [1536, 384]
  fc1Bias: Float32Array; // [1536]
  fc2Weight: Float32Array; // [384, 1536]
  fc2Bias: Float32Array; // [384]
}

export interface ShardAWeights {
  role: 'node-a';
  rawBuffer: Float32Array; // Length: 5,618,304
  checksum: string;
  tokEmb: Float32Array; // [256, 384] = 98,304
  posEmb: Float32Array; // [512, 384] = 196,608
  layers: [TransformerLayerWeights, TransformerLayerWeights, TransformerLayerWeights]; // Layers 0, 1, 2
}

export interface ShardBWeights {
  role: 'node-b';
  rawBuffer: Float32Array; // Length: 5,422,464
  checksum: string;
  layers: [TransformerLayerWeights, TransformerLayerWeights, TransformerLayerWeights]; // Layers 3, 4, 5
  lnFWeight: Float32Array; // [384]
  lnFBias: Float32Array; // [384]
  lmHeadWeight: Float32Array; // [256, 384] = 98,304
}

export interface LayerKVCache {
  k: Float32Array; // [maxSeqLen, dModel] = [512, 384]
  v: Float32Array; // [maxSeqLen, dModel] = [512, 384]
  seqLen: number;
}

/**
 * Carve a single layer's weight views from a contiguous Float32Array at `offset`.
 * Advances and returns the new offset (always +1,774,464 floats).
 */
function bindLayerViews(
  buffer: Float32Array,
  offset: number,
  layerIndex: number
): { layer: TransformerLayerWeights; nextOffset: number } {
  const D = MODEL_SPEC.dModel; // 384
  const FF = MODEL_SPEC.dFf; // 1536
  let ptr = offset;

  const slice = (len: number) => {
    const view = buffer.subarray(ptr, ptr + len);
    ptr += len;
    return view;
  };

  const ln1Weight = slice(D);
  const ln1Bias = slice(D);
  const qWeight = slice(D * D);
  const qBias = slice(D);
  const kWeight = slice(D * D);
  const kBias = slice(D);
  const vWeight = slice(D * D);
  const vBias = slice(D);
  const outWeight = slice(D * D);
  const outBias = slice(D);
  const ln2Weight = slice(D);
  const ln2Bias = slice(D);
  const fc1Weight = slice(FF * D);
  const fc1Bias = slice(FF);
  const fc2Weight = slice(D * FF);
  const fc2Bias = slice(D);

  return {
    layer: {
      layerIndex,
      ln1Weight,
      ln1Bias,
      qWeight,
      qBias,
      kWeight,
      kBias,
      vWeight,
      vBias,
      outWeight,
      outBias,
      ln2Weight,
      ln2Bias,
      fc1Weight,
      fc1Bias,
      fc2Weight,
      fc2Bias,
    },
    nextOffset: ptr,
  };
}

/**
 * Populate a layer's weights using the deterministic seed-42 PRNG.
 * To match standard PyTorch module initialization with `torch.manual_seed(42)`,
 * every parameter slot consumes one normal sample from the PRNG, with LayerNorm
 * scale weights centered around 1.0 and biases scaled by 0.01.
 */
function initLayerWithPrng(layer: TransformerLayerWeights, prng: TorchMT19937): void {
  const fillCentered = (arr: Float32Array, mean: number, std: number) => {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = prng.nextNormal(mean, std);
    }
  };

  fillCentered(layer.ln1Weight, 1.0, 0.01);
  fillCentered(layer.ln1Bias, 0.0, 0.005);
  prng.fillNormal(layer.qWeight, 0.02);
  fillCentered(layer.qBias, 0.0, 0.005);
  prng.fillNormal(layer.kWeight, 0.02);
  fillCentered(layer.kBias, 0.0, 0.005);
  prng.fillNormal(layer.vWeight, 0.02);
  fillCentered(layer.vBias, 0.0, 0.005);
  prng.fillNormal(layer.outWeight, 0.02 / Math.sqrt(2 * MODEL_SPEC.nLayers));
  fillCentered(layer.outBias, 0.0, 0.005);
  fillCentered(layer.ln2Weight, 1.0, 0.01);
  fillCentered(layer.ln2Bias, 0.0, 0.005);
  prng.fillNormal(layer.fc1Weight, 0.02);
  fillCentered(layer.fc1Bias, 0.0, 0.005);
  prng.fillNormal(layer.fc2Weight, 0.02 / Math.sqrt(2 * MODEL_SPEC.nLayers));
  fillCentered(layer.fc2Bias, 0.0, 0.005);
}

/**
 * Initialize Shard A (Phone 1: Embeddings + Layers 0..2 = 5,618,304 FP32 parameters)
 * Either from deterministic PyTorch Seed 42 PRNG or from an external binary buffer.
 */
export function createShardAWeights(
  seed = MODEL_SPEC.seed,
  externalBuffer?: Float32Array
): ShardAWeights {
  const rawBuffer =
    externalBuffer && externalBuffer.length === MODEL_SPEC.shardAParams
      ? externalBuffer
      : new Float32Array(MODEL_SPEC.shardAParams);

  let ptr = 0;
  const tokEmb = rawBuffer.subarray(ptr, ptr + MODEL_SPEC.tokEmbParams);
  ptr += MODEL_SPEC.tokEmbParams;
  const posEmb = rawBuffer.subarray(ptr, ptr + MODEL_SPEC.posEmbParams);
  ptr += MODEL_SPEC.posEmbParams;

  const l0 = bindLayerViews(rawBuffer, ptr, 0);
  const l1 = bindLayerViews(rawBuffer, l0.nextOffset, 1);
  const l2 = bindLayerViews(rawBuffer, l1.nextOffset, 2);

  if (!externalBuffer) {
    const prng = new TorchMT19937(seed);
    prng.fillNormal(tokEmb, 0.02);
    prng.fillNormal(posEmb, 0.02);
    initLayerWithPrng(l0.layer, prng);
    initLayerWithPrng(l1.layer, prng);
    initLayerWithPrng(l2.layer, prng);
  }

  return {
    role: 'node-a',
    rawBuffer,
    checksum: computeTensorChecksum(rawBuffer),
    tokEmb,
    posEmb,
    layers: [l0.layer, l1.layer, l2.layer],
  };
}

/**
 * Initialize Shard B (Phone 2: Layers 3..5 + Final LayerNorm + LM Head = 5,422,464 FP32 parameters)
 * Advances the PRNG past Shard A's 5,618,304 parameters so Phone 2 never allocates Shard A.
 */
export function createShardBWeights(
  seed = MODEL_SPEC.seed,
  externalBuffer?: Float32Array
): ShardBWeights {
  const rawBuffer =
    externalBuffer && externalBuffer.length === MODEL_SPEC.shardBParams
      ? externalBuffer
      : new Float32Array(MODEL_SPEC.shardBParams);

  let ptr = 0;
  const l3 = bindLayerViews(rawBuffer, ptr, 3);
  const l4 = bindLayerViews(rawBuffer, l3.nextOffset, 4);
  const l5 = bindLayerViews(rawBuffer, l4.nextOffset, 5);
  ptr = l5.nextOffset;

  const lnFWeight = rawBuffer.subarray(ptr, ptr + MODEL_SPEC.dModel);
  ptr += MODEL_SPEC.dModel;
  const lnFBias = rawBuffer.subarray(ptr, ptr + MODEL_SPEC.dModel);
  ptr += MODEL_SPEC.dModel;
  const lmHeadWeight = rawBuffer.subarray(ptr, ptr + MODEL_SPEC.lmHeadParams);
  ptr += MODEL_SPEC.lmHeadParams;

  if (!externalBuffer) {
    const prng = new TorchMT19937(seed);
    // Deterministically skip Shard A's exact 5,618,304 parameters
    prng.skipNormals(MODEL_SPEC.shardAParams);
    initLayerWithPrng(l3.layer, prng);
    initLayerWithPrng(l4.layer, prng);
    initLayerWithPrng(l5.layer, prng);
    for (let i = 0; i < MODEL_SPEC.dModel; i++) {
      lnFWeight[i] = prng.nextNormal(1.0, 0.01);
    }
    for (let i = 0; i < MODEL_SPEC.dModel; i++) {
      lnFBias[i] = prng.nextNormal(0.0, 0.005);
    }
    prng.fillNormal(lmHeadWeight, 0.02);
  }

  return {
    role: 'node-b',
    rawBuffer,
    checksum: computeTensorChecksum(rawBuffer),
    layers: [l3.layer, l4.layer, l5.layer],
    lnFWeight,
    lnFBias,
    lmHeadWeight,
  };
}

/**
 * LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 * Operates on a single vector of length D = 384 at `xOffset`.
 */
export function layerNormInPlace(
  out: Float32Array,
  outOffset: number,
  x: Float32Array,
  xOffset: number,
  weight: Float32Array,
  bias: Float32Array,
  d = MODEL_SPEC.dModel
): void {
  let sum = 0.0;
  for (let i = 0; i < d; i++) {
    sum += x[xOffset + i];
  }
  const mean = sum / d;
  let sqSum = 0.0;
  for (let i = 0; i < d; i++) {
    const diff = x[xOffset + i] - mean;
    sqSum += diff * diff;
  }
  const invStd = 1.0 / Math.sqrt(sqSum / d + MODEL_SPEC.layerNormEps);
  for (let i = 0; i < d; i++) {
    out[outOffset + i] = (x[xOffset + i] - mean) * invStd * weight[i] + bias[i];
  }
}

/**
 * Fast 4x-unrolled Matrix-Vector multiplication:
 * out[0..outDim-1] = weight[outDim, inDim] * input[inOffset..inOffset+inDim-1] + bias[0..outDim-1]
 */
export function matVecMul(
  out: Float32Array,
  outOffset: number,
  input: Float32Array,
  inOffset: number,
  weight: Float32Array,
  bias: Float32Array | null,
  outDim: number,
  inDim: number
): void {
  for (let row = 0; row < outDim; row++) {
    const wRowOffset = row * inDim;
    let acc = bias ? bias[row] : 0.0;
    let col = 0;
    const limit = inDim - 3;
    for (; col < limit; col += 4) {
      acc +=
        weight[wRowOffset + col] * input[inOffset + col] +
        weight[wRowOffset + col + 1] * input[inOffset + col + 1] +
        weight[wRowOffset + col + 2] * input[inOffset + col + 2] +
        weight[wRowOffset + col + 3] * input[inOffset + col + 3];
    }
    for (; col < inDim; col++) {
      acc += weight[wRowOffset + col] * input[inOffset + col];
    }
    out[outOffset + row] = Math.fround(acc);
  }
}

/**
 * Exact Hendrycks & Gimpel GELU activation (tanh approximation matching GPT-2 / PyTorch GELU):
 * 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 */
const SQRT_2_OVER_PI = 0.7978845608028654;
export function geluInPlace(x: Float32Array, offset: number, len: number): void {
  for (let i = 0; i < len; i++) {
    const v = x[offset + i];
    const inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
    x[offset + i] = Math.fround(0.5 * v * (1.0 + Math.tanh(inner)));
  }
}

/**
 * Pre-allocated scratch buffers for a 3-layer shard to guarantee zero GC pause during token generation.
 */
export class TransformerShardScratch {
  public readonly normBuf = new Float32Array(MODEL_SPEC.dModel);
  public readonly qBuf = new Float32Array(MODEL_SPEC.dModel);
  public readonly attnOutBuf = new Float32Array(MODEL_SPEC.dModel);
  public readonly projOutBuf = new Float32Array(MODEL_SPEC.dModel);
  public readonly ffnHiddenBuf = new Float32Array(MODEL_SPEC.dFf);
  public readonly scoresBuf = new Float32Array(MODEL_SPEC.maxSeqLen);
}

/**
 * Execute a single Transformer Decoder block at sequence step `pos` using KV cache.
 * - `hidden`: Float32Array of length `dModel` (384) representing current token hidden state at `pos`
 * - Updates `hidden` in-place with residual connection after Attention and MLP.
 */
export function runTransformerLayerStep(
  hidden: Float32Array,
  hiddenOffset: number,
  pos: number,
  layer: TransformerLayerWeights,
  kvCache: LayerKVCache,
  scratch: TransformerShardScratch
): void {
  const D = MODEL_SPEC.dModel; // 384
  const H = MODEL_SPEC.nHeads; // 6
  const DH = MODEL_SPEC.dHead; // 64
  const FF = MODEL_SPEC.dFf; // 1536
  const scale = 1.0 / Math.sqrt(DH); // 1 / 8 = 0.125

  // 1. Pre-LayerNorm 1
  layerNormInPlace(scratch.normBuf, 0, hidden, hiddenOffset, layer.ln1Weight, layer.ln1Bias, D);

  // 2. Q, K, V projections for current position `pos`
  const kvOffset = pos * D;
  matVecMul(scratch.qBuf, 0, scratch.normBuf, 0, layer.qWeight, layer.qBias, D, D);
  matVecMul(kvCache.k, kvOffset, scratch.normBuf, 0, layer.kWeight, layer.kBias, D, D);
  matVecMul(kvCache.v, kvOffset, scratch.normBuf, 0, layer.vWeight, layer.vBias, D, D);
  kvCache.seqLen = pos + 1;

  // 3. Causal Multi-Head Self-Attention (6 heads, head_dim = 64) over positions 0..pos
  scratch.attnOutBuf.fill(0);
  for (let h = 0; h < H; h++) {
    const headOffset = h * DH;
    let maxScore = -Infinity;

    // Compute scaled dot-product attention scores Q_h · K_h[t] / sqrt(64) for t in 0..pos
    for (let t = 0; t <= pos; t++) {
      const kBase = t * D + headOffset;
      let dot = 0.0;
      for (let d = 0; d < DH; d += 4) {
        dot +=
          scratch.qBuf[headOffset + d] * kvCache.k[kBase + d] +
          scratch.qBuf[headOffset + d + 1] * kvCache.k[kBase + d + 1] +
          scratch.qBuf[headOffset + d + 2] * kvCache.k[kBase + d + 2] +
          scratch.qBuf[headOffset + d + 3] * kvCache.k[kBase + d + 3];
      }
      const score = dot * scale;
      scratch.scoresBuf[t] = score;
      if (score > maxScore) maxScore = score;
    }

    // Softmax over 0..pos
    let sumExp = 0.0;
    for (let t = 0; t <= pos; t++) {
      const e = Math.exp(scratch.scoresBuf[t] - maxScore);
      scratch.scoresBuf[t] = e;
      sumExp += e;
    }
    const invSumExp = 1.0 / sumExp;

    // Weighted sum of V_h[t]
    for (let t = 0; t <= pos; t++) {
      const prob = scratch.scoresBuf[t] * invSumExp;
      const vBase = t * D + headOffset;
      for (let d = 0; d < DH; d += 4) {
        scratch.attnOutBuf[headOffset + d] += prob * kvCache.v[vBase + d];
        scratch.attnOutBuf[headOffset + d + 1] += prob * kvCache.v[vBase + d + 1];
        scratch.attnOutBuf[headOffset + d + 2] += prob * kvCache.v[vBase + d + 2];
        scratch.attnOutBuf[headOffset + d + 3] += prob * kvCache.v[vBase + d + 3];
      }
    }
  }

  // 4. Attention Output Projection + Residual 1
  matVecMul(scratch.projOutBuf, 0, scratch.attnOutBuf, 0, layer.outWeight, layer.outBias, D, D);
  for (let i = 0; i < D; i++) {
    hidden[hiddenOffset + i] = Math.fround(hidden[hiddenOffset + i] + scratch.projOutBuf[i]);
  }

  // 5. Pre-LayerNorm 2
  layerNormInPlace(scratch.normBuf, 0, hidden, hiddenOffset, layer.ln2Weight, layer.ln2Bias, D);

  // 6. Feed-Forward Network: 384 -> 1536 -> GELU -> 384 + Residual 2
  matVecMul(scratch.ffnHiddenBuf, 0, scratch.normBuf, 0, layer.fc1Weight, layer.fc1Bias, FF, D);
  geluInPlace(scratch.ffnHiddenBuf, 0, FF);
  matVecMul(scratch.projOutBuf, 0, scratch.ffnHiddenBuf, 0, layer.fc2Weight, layer.fc2Bias, D, FF);
  for (let i = 0; i < D; i++) {
    hidden[hiddenOffset + i] = Math.fround(hidden[hiddenOffset + i] + scratch.projOutBuf[i]);
  }
}

/**
 * Node A Runtime (Phone 1: Token Embedding + Positional Embedding + Layers 0, 1, 2)
 * Holds ONLY Shard A weights (5,618,304 params). Cannot compute Layers 3..5 or logits alone.
 */
export class NodeARuntime {
  public readonly shard: ShardAWeights;
  private readonly kvCaches: [LayerKVCache, LayerKVCache, LayerKVCache];
  private readonly scratch = new TransformerShardScratch();

  constructor(shard?: ShardAWeights) {
    this.shard = shard ?? createShardAWeights(MODEL_SPEC.seed);
    this.kvCaches = [this.createCache(), this.createCache(), this.createCache()];
  }

  private createCache(): LayerKVCache {
    return {
      k: new Float32Array(MODEL_SPEC.maxSeqLen * MODEL_SPEC.dModel),
      v: new Float32Array(MODEL_SPEC.maxSeqLen * MODEL_SPEC.dModel),
      seqLen: 0,
    };
  }

  public resetCache(): void {
    for (const c of this.kvCaches) {
      c.seqLen = 0;
    }
  }

  /**
   * Forward pass for a single token at sequence position `pos` (KV-cached pipeline).
   * Returns a Float32Array(384) representing Layer 2 output activations for `pos`.
   */
  public forwardStep(byteToken: number, pos: number): Float32Array {
    if (pos < 0 || pos >= MODEL_SPEC.maxSeqLen) {
      throw new RangeError(`Position ${pos} out of bounds [0, ${MODEL_SPEC.maxSeqLen - 1}]`);
    }
    const b = byteToken & 0xff;
    const D = MODEL_SPEC.dModel;
    const hidden = new Float32Array(D);

    const tokBase = b * D;
    const posBase = pos * D;
    for (let i = 0; i < D; i++) {
      hidden[i] = Math.fround(this.shard.tokEmb[tokBase + i] + this.shard.posEmb[posBase + i]);
    }

    for (let l = 0; l < 3; l++) {
      runTransformerLayerStep(hidden, 0, pos, this.shard.layers[l], this.kvCaches[l], this.scratch);
    }

    return hidden;
  }

  /**
   * Full-sequence forward pass for `tokens` of length `T` (1 <= T <= 512).
   * Recomputes all positions 0..T-1 and returns a contiguous Float32Array(T * 384)
   * (or padded to [512 * 384] if `padToMaxSeqLen` is true).
   */
  public forwardFullSequence(
    tokens: Uint8Array | number[],
    padToMaxSeqLen = false
  ): Float32Array {
    const T = Math.min(tokens.length, MODEL_SPEC.maxSeqLen);
    const D = MODEL_SPEC.dModel;
    const outLen = padToMaxSeqLen ? MODEL_SPEC.maxSeqLen * D : T * D;
    const outTensor = new Float32Array(outLen);

    this.resetCache();
    for (let pos = 0; pos < T; pos++) {
      const b = tokens[pos] & 0xff;
      const tokBase = b * D;
      const posBase = pos * D;
      const outBase = pos * D;
      for (let i = 0; i < D; i++) {
        outTensor[outBase + i] = Math.fround(
          this.shard.tokEmb[tokBase + i] + this.shard.posEmb[posBase + i]
        );
      }
      for (let l = 0; l < 3; l++) {
        runTransformerLayerStep(
          outTensor,
          outBase,
          pos,
          this.shard.layers[l],
          this.kvCaches[l],
          this.scratch
        );
      }
    }
    return outTensor;
  }
}

/**
 * Node B Runtime (Phone 2: Layers 3, 4, 5 + Final LayerNorm + LM Head [384 -> 256])
 * Holds ONLY Shard B weights (5,422,464 params). Receives Layer 2 activations from Phone 1.
 */
export class NodeBRuntime {
  public readonly shard: ShardBWeights;
  private readonly kvCaches: [LayerKVCache, LayerKVCache, LayerKVCache];
  private readonly scratch = new TransformerShardScratch();

  constructor(shard?: ShardBWeights) {
    this.shard = shard ?? createShardBWeights(MODEL_SPEC.seed);
    this.kvCaches = [this.createCache(), this.createCache(), this.createCache()];
  }

  private createCache(): LayerKVCache {
    return {
      k: new Float32Array(MODEL_SPEC.maxSeqLen * MODEL_SPEC.dModel),
      v: new Float32Array(MODEL_SPEC.maxSeqLen * MODEL_SPEC.dModel),
      seqLen: 0,
    };
  }

  public resetCache(): void {
    for (const c of this.kvCaches) {
      c.seqLen = 0;
    }
  }

  /**
   * Forward pass on Phone 2 for a single step `pos` given Layer 2 activation vector [384] from Phone 1.
   * Returns both the [256] logits vector and the final normalized hidden state [384].
   */
  public forwardStep(
    layer2Activation: Float32Array,
    pos: number
  ): { logits: Float32Array; finalHidden: Float32Array } {
    const D = MODEL_SPEC.dModel;
    const V = MODEL_SPEC.vocabSize;
    const hidden = new Float32Array(D);
    hidden.set(layer2Activation.subarray(0, D));

    for (let l = 0; l < 3; l++) {
      runTransformerLayerStep(hidden, 0, pos, this.shard.layers[l], this.kvCaches[l], this.scratch);
    }

    const finalHidden = new Float32Array(D);
    layerNormInPlace(
      finalHidden,
      0,
      hidden,
      0,
      this.shard.lnFWeight,
      this.shard.lnFBias,
      D
    );

    const logits = new Float32Array(V);
    matVecMul(logits, 0, finalHidden, 0, this.shard.lmHeadWeight, null, V, D);

    return { logits, finalHidden };
  }

  /**
   * Full-sequence forward pass on Phone 2 given a [T * 384] (or [512 * 384]) activation tensor from Phone 1.
   * Returns the [256] logits for the last active sequence position `seqLen - 1`.
   */
  public forwardFullSequence(
    layer2Activations: Float32Array,
    seqLen: number
  ): { logits: Float32Array; finalHidden: Float32Array } {
    const D = MODEL_SPEC.dModel;
    const V = MODEL_SPEC.vocabSize;
    const T = Math.min(seqLen, MODEL_SPEC.maxSeqLen);
    this.resetCache();

    const stepHidden = new Float32Array(D);
    for (let pos = 0; pos < T; pos++) {
      const base = pos * D;
      stepHidden.set(layer2Activations.subarray(base, base + D));
      for (let l = 0; l < 3; l++) {
        runTransformerLayerStep(
          stepHidden,
          0,
          pos,
          this.shard.layers[l],
          this.kvCaches[l],
          this.scratch
        );
      }
    }

    const finalHidden = new Float32Array(D);
    layerNormInPlace(
      finalHidden,
      0,
      stepHidden,
      0,
      this.shard.lnFWeight,
      this.shard.lnFBias,
      D
    );

    const logits = new Float32Array(V);
    matVecMul(logits, 0, finalHidden, 0, this.shard.lmHeadWeight, null, V, D);

    return { logits, finalHidden };
  }
}

/**
 * Deterministic token selection:
 * - At temperature <= 0.0: strict greedy argmax over 256 byte logits (100% deterministic)
 * - At temperature > 0.0: temperature-scaled softmax sampling using seeded PRNG
 */
export function sampleNextByte(
  logits: Float32Array,
  temperature = 0.0,
  prng?: TorchMT19937
): { nextByte: number; topK: Array<{ byte: number; logit: number; prob: number }> } {
  const V = MODEL_SPEC.vocabSize;

  // Compute softmax probabilities (using temp=1.0 when temperature=0.0 for display diagnostics)
  const scaleTemp = temperature > 1e-5 ? temperature : 1.0;
  let maxLogit = -Infinity;
  let argmaxIdx = 0;
  for (let i = 0; i < V; i++) {
    const val = logits[i];
    if (val > maxLogit) {
      maxLogit = val;
      argmaxIdx = i;
    }
  }

  const probs = new Float32Array(V);
  let sumExp = 0.0;
  for (let i = 0; i < V; i++) {
    const e = Math.exp((logits[i] - maxLogit) / scaleTemp);
    probs[i] = e;
    sumExp += e;
  }
  for (let i = 0; i < V; i++) {
    probs[i] /= sumExp;
  }

  // Extract top 5 candidates for live telemetry inspection
  const indices = Array.from({ length: V }, (_, i) => i);
  indices.sort((a, b) => logits[b] - logits[a]);
  const topK = indices.slice(0, 5).map((idx) => ({
    byte: idx,
    logit: logits[idx],
    prob: probs[idx],
  }));

  if (temperature <= 1e-5) {
    return { nextByte: argmaxIdx, topK };
  }

  const r = prng ? prng.nextFloat() : Math.random();
  let cdf = 0.0;
  for (let i = 0; i < V; i++) {
    cdf += probs[i];
    if (r <= cdf) {
      return { nextByte: i, topK };
    }
  }
  return { nextByte: argmaxIdx, topK };
}

/**
 * Format a byte (0..255) into a human-readable glyph for telemetry tables
 */
export function formatByteGlyph(byte: number): string {
  const b = byte & 0xff;
  if (b === 0x0a) return '\\n';
  if (b === 0x0d) return '\\r';
  if (b === 0x09) return '\\t';
  if (b === 0x20) return '␣';
  if (b >= 33 && b <= 126) return String.fromCharCode(b);
  return `0x${b.toString(16).toUpperCase().padStart(2, '0')}`;
}
