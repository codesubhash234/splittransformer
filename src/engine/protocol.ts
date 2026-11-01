/**
 * Binary Activation Serialization & Streaming Protocol over WebRTC RTCDataChannel
 *
 * Supports:
 * 1. Direct raw binary messages (as described in the hackathon specification):
 *    - Phone 1 -> Phone 2: `Float32Array` buffer of Layer 2 activations
 *    - Phone 2 -> Phone 1: `Uint8Array([nextByte])` 1-byte token buffer
 * 2. Framed binary SCTP packets:
 *    - Single-step KV-cached FP32 `[1, 384]` (1,536 B payload)
 *    - Coalesced Prompt Prefill Batch `[L, 384]` (1-trip prompt prefill)
 *    - Dynamic Symmetric Int8 Quantized `[1, 384 Int8 + FP32 Scale]` (388 B payload — 75% bandwidth reduction)
 *    - Multi-chunk SCTP `[1, T, 384]` full-sequence tensor transport
 */

export const PACKET_MAGIC = 0x5342; // 'SB' (SplitByte)
export const MAX_SCTP_PAYLOAD_BYTES = 16384; // 16 KB safe for all iOS Safari & Android Chrome WebRTC DataChannels

export enum PacketType {
  RESET_SESSION = 0x01,
  ACTIVATION_STEP = 0x02, // KV-cached single-position [1, 384] Float32 activation
  ACTIVATION_CHUNK = 0x03, // Chunked full-sequence [1, T, 384] or [1, 512, 384] Float32 activation
  TOKEN_RESULT = 0x04, // Next byte [0..255] + optional top-5 logits & Node B compute ms
  PING = 0x05,
  PONG = 0x06,
  REMOTE_START_REQUEST = 0x07, // Phone 2 requests Phone 1 to start split inference
  PREFILL_BATCH = 0x08, // Coalesced multi-position prompt prefill [count, 384] Float32 activations
  ACTIVATION_STEP_INT8 = 0x09, // Dynamic symmetric Int8 quantized [1, 384] activation + FP32 scale (404 B total)
}

export interface DecodedActivationStep {
  type: PacketType.ACTIVATION_STEP;
  stepId: number;
  seqPos: number;
  inputByte: number;
  temperature: number;
  wireBytes: number;
  activations: Float32Array; // [384]
}

export interface DecodedPrefillBatch {
  type: PacketType.PREFILL_BATCH;
  count: number;
  activations: Float32Array; // [count * 384]
}

export interface DecodedActivationFull {
  type: PacketType.ACTIVATION_CHUNK;
  stepId: number;
  seqLen: number;
  inputByte: number;
  temperature: number;
  activations: Float32Array; // [seqLen * 384] or [512 * 384]
}

export interface DecodedTokenResult {
  type: PacketType.TOKEN_RESULT;
  stepId: number;
  seqPos: number;
  nextByte: number;
  nodeBComputeMs: number;
  topK: Array<{ byte: number; logit: number; prob: number }>;
}

export interface DecodedResetSession {
  type: PacketType.RESET_SESSION;
  sessionId: number;
  targetNewBytes: number;
  promptBytes: number[];
}

export interface DecodedRemoteStartRequest {
  type: PacketType.REMOTE_START_REQUEST;
}

export interface DecodedPingPong {
  type: PacketType.PING | PacketType.PONG;
  timestampMs: number;
}

export type DecodedPacket =
  | DecodedActivationStep
  | DecodedPrefillBatch
  | DecodedActivationFull
  | DecodedTokenResult
  | DecodedResetSession
  | DecodedRemoteStartRequest
  | DecodedPingPong;

/**
 * Encode a single-step Layer 2 activation tensor [384] into a compact binary ArrayBuffer (1,552 bytes total:
 * 16-byte aligned header + 1,536 bytes of Float32 activations).
 */
export function encodeActivationStepPacket(
  stepId: number,
  seqPos: number,
  temperature: number,
  activations: Float32Array,
  inputByte = 0
): ArrayBuffer {
  const headerBytes = 16;
  const payloadBytes = activations.byteLength;
  const buf = new ArrayBuffer(headerBytes + payloadBytes);
  const view = new DataView(buf);

  view.setUint16(0, PACKET_MAGIC, true);
  view.setUint8(2, PacketType.ACTIVATION_STEP);
  view.setUint8(3, inputByte & 0xff);
  view.setUint32(4, stepId >>> 0, true);
  view.setUint32(8, seqPos >>> 0, true);
  view.setFloat32(12, temperature, true);

  const floatDest = new Float32Array(buf, headerBytes, activations.length);
  floatDest.set(activations);
  return buf;
}

/**
 * Encode a single-step Layer 2 activation tensor [384] using Dynamic Symmetric Int8 Quantization:
 * - 20-byte header (includes 4-byte FP32 dequantization scale) + 384 bytes Int8 payload = 404 bytes total!
 * - Cuts wireless DataChannel payload size by 74% (388 B vs 1,536 B).
 */
export function encodeActivationStepInt8Packet(
  stepId: number,
  seqPos: number,
  temperature: number,
  activations: Float32Array,
  inputByte = 0
): ArrayBuffer {
  const len = activations.length; // 384
  let maxAbs = 1e-8;
  for (let i = 0; i < len; i++) {
    const a = Math.abs(activations[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs / 127.0;
  const invScale = 127.0 / maxAbs;

  const headerBytes = 20;
  const buf = new ArrayBuffer(headerBytes + len);
  const view = new DataView(buf);

  view.setUint16(0, PACKET_MAGIC, true);
  view.setUint8(2, PacketType.ACTIVATION_STEP_INT8);
  view.setUint8(3, inputByte & 0xff);
  view.setUint32(4, stepId >>> 0, true);
  view.setUint32(8, seqPos >>> 0, true);
  view.setFloat32(12, temperature, true);
  view.setFloat32(16, scale, true);

  const int8Dest = new Int8Array(buf, headerBytes, len);
  for (let i = 0; i < len; i++) {
    int8Dest[i] = Math.max(-127, Math.min(127, Math.round(activations[i] * invScale)));
  }
  return buf;
}

/**
 * Coalesce all prompt prefill activations `0 .. count-1` into a single binary packet
 * so Phone 1 -> Phone 2 prefill happens in 1 DataChannel message instead of `count` round-trips.
 */
export function encodePrefillBatchPacket(
  count: number,
  activations: Float32Array
): ArrayBuffer {
  const headerBytes = 8;
  const buf = new ArrayBuffer(headerBytes + activations.byteLength);
  const view = new DataView(buf);

  view.setUint16(0, PACKET_MAGIC, true);
  view.setUint8(2, PacketType.PREFILL_BATCH);
  view.setUint8(3, 0);
  view.setUint32(4, count >>> 0, true);

  new Float32Array(buf, headerBytes, activations.length).set(activations);
  return buf;
}

/**
 * Encode a full-sequence Layer 2 activation tensor [T * 384] or [512 * 384] into one or more
 * SCTP-safe binary ArrayBuffer chunks (each <= 16 KB).
 */
export function encodeActivationFullChunks(
  stepId: number,
  seqLen: number,
  temperature: number,
  activations: Float32Array,
  inputByte = 0
): ArrayBuffer[] {
  const rawBytes = new Uint8Array(
    activations.buffer,
    activations.byteOffset,
    activations.byteLength
  );
  const totalBytes = rawBytes.byteLength;
  const totalChunks = Math.max(1, Math.ceil(totalBytes / MAX_SCTP_PAYLOAD_BYTES));
  const chunks: ArrayBuffer[] = [];

  for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    const start = chunkIdx * MAX_SCTP_PAYLOAD_BYTES;
    const end = Math.min(start + MAX_SCTP_PAYLOAD_BYTES, totalBytes);
    const chunkLen = end - start;
    const headerBytes = 24;
    const buf = new ArrayBuffer(headerBytes + chunkLen);
    const view = new DataView(buf);

    view.setUint16(0, PACKET_MAGIC, true);
    view.setUint8(2, PacketType.ACTIVATION_CHUNK);
    view.setUint8(3, inputByte & 0xff);
    view.setUint32(4, stepId >>> 0, true);
    view.setUint16(8, seqLen, true);
    view.setUint16(10, totalChunks, true);
    view.setUint16(12, chunkIdx, true);
    view.setUint16(14, 0);
    view.setUint32(16, totalBytes >>> 0, true);
    view.setFloat32(20, temperature, true);

    new Uint8Array(buf, headerBytes, chunkLen).set(rawBytes.subarray(start, end));
    chunks.push(buf);
  }

  return chunks;
}

/**
 * Encode Phone 2's generated byte result.
 * Supports both `rawSingleByte = true` (exact 1-byte Uint8Array([nextByte]).buffer)
 * and `rawSingleByte = false` (framed binary response with Node B compute latency & top-5 logits).
 */
export function encodeTokenResultPacket(
  stepId: number,
  seqPos: number,
  nextByte: number,
  nodeBComputeMs: number,
  topK: Array<{ byte: number; logit: number; prob: number }>,
  rawSingleByte = false
): ArrayBuffer {
  if (rawSingleByte) {
    const arr = new Uint8Array([nextByte & 0xff]);
    return arr.buffer;
  }

  const kCount = Math.min(5, topK.length);
  const headerBytes = 16;
  const itemBytes = 12; // uint32 byte + float32 logit + float32 prob
  const buf = new ArrayBuffer(headerBytes + kCount * itemBytes);
  const view = new DataView(buf);

  view.setUint16(0, PACKET_MAGIC, true);
  view.setUint8(2, PacketType.TOKEN_RESULT);
  view.setUint8(3, nextByte & 0xff);
  view.setUint32(4, stepId >>> 0, true);
  view.setUint32(8, seqPos >>> 0, true);
  view.setFloat32(12, nodeBComputeMs, true);

  for (let i = 0; i < kCount; i++) {
    const base = headerBytes + i * itemBytes;
    view.setUint32(base, topK[i].byte & 0xff, true);
    view.setFloat32(base + 4, topK[i].logit, true);
    view.setFloat32(base + 8, topK[i].prob, true);
  }

  return buf;
}

export function encodeResetSessionPacket(
  sessionId: number,
  promptBytes: number[] = [],
  targetNewBytes = 16
): ArrayBuffer {
  const headerBytes = 12;
  const buf = new ArrayBuffer(headerBytes + promptBytes.length);
  const view = new DataView(buf);
  view.setUint16(0, PACKET_MAGIC, true);
  view.setUint8(2, PacketType.RESET_SESSION);
  view.setUint8(3, targetNewBytes & 0xff);
  view.setUint32(4, sessionId >>> 0, true);
  view.setUint32(8, promptBytes.length >>> 0, true);
  const u8 = new Uint8Array(buf, headerBytes, promptBytes.length);
  for (let i = 0; i < promptBytes.length; i++) {
    u8[i] = promptBytes[i] & 0xff;
  }
  return buf;
}

export function encodeRemoteStartRequestPacket(): ArrayBuffer {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setUint16(0, PACKET_MAGIC, true);
  view.setUint8(2, PacketType.REMOTE_START_REQUEST);
  view.setUint8(3, 0);
  return buf;
}

/**
 * Reassembler and decoder for incoming RTCDataChannel binary ArrayBuffers.
 */
export class DataChannelPacketAssembler {
  private pendingStepId = -1;
  private pendingSeqLen = 0;
  private pendingInputByte = 0;
  private pendingTemp = 0.0;
  private pendingTotalChunks = 0;
  private pendingReceivedChunks = 0;
  private pendingBuffer: Uint8Array | null = null;

  public decode(buffer: ArrayBuffer): DecodedPacket | null {
    // 1. Raw 1-byte token buffer: `new Uint8Array([nextByte]).buffer`
    if (buffer.byteLength === 1) {
      const b = new Uint8Array(buffer)[0];
      return {
        type: PacketType.TOKEN_RESULT,
        stepId: 0,
        seqPos: 0,
        nextByte: b,
        nodeBComputeMs: 0,
        topK: [{ byte: b, logit: 1.0, prob: 1.0 }],
      };
    }

    const view = new DataView(buffer);
    const magic = buffer.byteLength >= 2 ? view.getUint16(0, true) : 0;

    // 2. If not prefixed with PACKET_MAGIC, treat as raw Float32Array activation buffer
    if (magic !== PACKET_MAGIC) {
      const floats = new Float32Array(buffer);
      if (floats.length === 384) {
        return {
          type: PacketType.ACTIVATION_STEP,
          stepId: 0,
          seqPos: 0,
          inputByte: 0,
          temperature: 0.0,
          wireBytes: buffer.byteLength,
          activations: floats,
        };
      }
      return {
        type: PacketType.ACTIVATION_CHUNK,
        stepId: 0,
        seqLen: Math.floor(floats.length / 384),
        inputByte: 0,
        temperature: 0.0,
        activations: floats,
      };
    }

    const pktType = view.getUint8(2) as PacketType;

    switch (pktType) {
      case PacketType.REMOTE_START_REQUEST: {
        return { type: PacketType.REMOTE_START_REQUEST };
      }

      case PacketType.RESET_SESSION: {
        const targetNewBytes = view.getUint8(3) || 16;
        const sessionId = view.getUint32(4, true);
        const promptLen = buffer.byteLength >= 12 ? view.getUint32(8, true) : 0;
        const promptBytes: number[] = [];
        if (promptLen > 0 && buffer.byteLength >= 12 + promptLen) {
          const u8 = new Uint8Array(buffer, 12, promptLen);
          for (let i = 0; i < promptLen; i++) promptBytes.push(u8[i]);
        }
        return {
          type: PacketType.RESET_SESSION,
          sessionId,
          targetNewBytes,
          promptBytes,
        };
      }

      case PacketType.PREFILL_BATCH: {
        const count = view.getUint32(4, true);
        const floatLen = (buffer.byteLength - 8) / 4;
        const activations = new Float32Array(floatLen);
        activations.set(new Float32Array(buffer, 8, floatLen));
        return {
          type: PacketType.PREFILL_BATCH,
          count,
          activations,
        };
      }

      case PacketType.ACTIVATION_STEP: {
        const inputByte = view.getUint8(3);
        const stepId = view.getUint32(4, true);
        const seqPos = view.getUint32(8, true);
        const temperature = view.getFloat32(12, true);
        const floatLen = (buffer.byteLength - 16) / 4;
        const activations = new Float32Array(floatLen);
        activations.set(new Float32Array(buffer, 16, floatLen));
        return {
          type: PacketType.ACTIVATION_STEP,
          stepId,
          seqPos,
          inputByte,
          temperature,
          wireBytes: buffer.byteLength - 16,
          activations,
        };
      }

      case PacketType.ACTIVATION_STEP_INT8: {
        const inputByte = view.getUint8(3);
        const stepId = view.getUint32(4, true);
        const seqPos = view.getUint32(8, true);
        const temperature = view.getFloat32(12, true);
        const scale = view.getFloat32(16, true);
        const len = buffer.byteLength - 20;
        const int8View = new Int8Array(buffer, 20, len);
        const activations = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          activations[i] = Math.fround(int8View[i] * scale);
        }
        return {
          type: PacketType.ACTIVATION_STEP,
          stepId,
          seqPos,
          inputByte,
          temperature,
          wireBytes: len + 4, // 388 B (384 Int8 + 4B scale)
          activations,
        };
      }

      case PacketType.ACTIVATION_CHUNK: {
        const inputByte = view.getUint8(3);
        const stepId = view.getUint32(4, true);
        const seqLen = view.getUint16(8, true);
        const totalChunks = view.getUint16(10, true);
        const chunkIdx = view.getUint16(12, true);
        const totalBytes = view.getUint32(16, true);
        const temperature = view.getFloat32(20, true);

        if (this.pendingStepId !== stepId || !this.pendingBuffer) {
          this.pendingStepId = stepId;
          this.pendingSeqLen = seqLen;
          this.pendingInputByte = inputByte;
          this.pendingTemp = temperature;
          this.pendingTotalChunks = totalChunks;
          this.pendingReceivedChunks = 0;
          this.pendingBuffer = new Uint8Array(totalBytes);
        }

        const chunkOffset = chunkIdx * MAX_SCTP_PAYLOAD_BYTES;
        const chunkBytes = new Uint8Array(buffer, 24);
        this.pendingBuffer.set(chunkBytes, chunkOffset);
        this.pendingReceivedChunks++;

        if (this.pendingReceivedChunks >= this.pendingTotalChunks) {
          const completedFloats = new Float32Array(
            this.pendingBuffer.buffer,
            this.pendingBuffer.byteOffset,
            this.pendingBuffer.byteLength / 4
          );
          const result: DecodedActivationFull = {
            type: PacketType.ACTIVATION_CHUNK,
            stepId,
            seqLen: this.pendingSeqLen,
            inputByte: this.pendingInputByte,
            temperature: this.pendingTemp,
            activations: completedFloats,
          };
          this.pendingBuffer = null;
          this.pendingStepId = -1;
          return result;
        }
        return null; // Wait for remaining chunks
      }

      case PacketType.TOKEN_RESULT: {
        const nextByte = view.getUint8(3);
        const stepId = view.getUint32(4, true);
        const seqPos = view.getUint32(8, true);
        const nodeBComputeMs = view.getFloat32(12, true);
        const kCount = Math.floor((buffer.byteLength - 16) / 12);
        const topK: Array<{ byte: number; logit: number; prob: number }> = [];
        for (let i = 0; i < kCount; i++) {
          const base = 16 + i * 12;
          topK.push({
            byte: view.getUint32(base, true),
            logit: view.getFloat32(base + 4, true),
            prob: view.getFloat32(base + 8, true),
          });
        }
        return {
          type: PacketType.TOKEN_RESULT,
          stepId,
          seqPos,
          nextByte,
          nodeBComputeMs,
          topK,
        };
      }

      case PacketType.PING:
      case PacketType.PONG: {
        return {
          type: pktType,
          timestampMs: view.getFloat64(8, true),
        };
      }

      default:
        return null;
    }
  }
}
