/**
 * Hand-Crafted WebAssembly (WASM) FP32 CPU Linear Algebra Kernel
 *
 * Compiles a self-contained WebAssembly binary module (`WebAssembly.Module` + `WebAssembly.Instance`)
 * directly in memory with zero external fetches and zero WebGPU/WebGL shaders.
 *
 * Exported WASM functions:
 * - `memory`: WebAssembly.Memory (1 page = 64 KB scratch arena for fast vector/head dot products)
 * - `dot_f32(aByteOffset: i32, bByteOffset: i32, count: i32) -> f32`
 */

const WASM_DOT_F32_BYTES = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, // "\0asm" magic
  0x01, 0x00, 0x00, 0x00, // version 1

  // Section 1: Type section (1 function signature: (i32, i32, i32) -> f32)
  0x01, 0x08, 0x01,
  0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7d,

  // Section 3: Function section (1 function of type 0)
  0x03, 0x02, 0x01, 0x00,

  // Section 5: Memory section (1 memory, min 1 page = 64KB)
  0x05, 0x03, 0x01, 0x00, 0x01,

  // Section 7: Export section ("memory" -> mem 0, "dot_f32" -> func 0)
  0x07, 0x13, 0x02,
  0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00,
  0x07, 0x64, 0x6f, 0x74, 0x5f, 0x66, 0x33, 0x32, 0x00, 0x00,

  // Section 10: Code section
  // func 0: params (0: i32 ptrA, 1: i32 ptrB, 2: i32 count), locals (3: f32 sum)
  0x0a, 0x2c, 0x01,
  0x2a, // body size = 42 bytes
  0x01, 0x01, 0x7d, // 1 local of type f32 (local 3 = sum)
  0x43, 0x00, 0x00, 0x00, 0x00, // f32.const 0
  0x21, 0x03, // local.set 3 (sum = 0.0)
  0x02, 0x40, // block
    0x03, 0x40, // loop
      0x20, 0x02, // local.get 2 (count)
      0x45, // i32.eqz
      0x0d, 0x01, // br_if 1 (break block if count == 0)

      0x20, 0x03, // local.get 3 (sum)
      0x20, 0x00, // local.get 0 (ptrA)
      0x2a, 0x02, 0x00, // f32.load align=4 offset=0
      0x20, 0x01, // local.get 1 (ptrB)
      0x2a, 0x02, 0x00, // f32.load align=4 offset=0
      0x94, // f32.mul
      0x92, // f32.add
      0x21, 0x03, // local.set 3

      0x20, 0x00, 0x41, 0x04, 0x6a, 0x21, 0x00, // ptrA += 4
      0x20, 0x01, 0x41, 0x04, 0x6a, 0x21, 0x01, // ptrB += 4
      0x20, 0x02, 0x41, 0x01, 0x6b, 0x21, 0x02, // count -= 1
      0x0c, 0x00, // br 0 (continue loop)
    0x0b, // end loop
  0x0b, // end block
  0x20, 0x03, // local.get 3 (return sum)
  0x0b, // end func
]);

export interface WasmCpuKernel {
  readonly isSupported: boolean;
  readonly memory: WebAssembly.Memory;
  readonly floatArena: Float32Array;
  dotF32: (aOffsetFloats: number, bOffsetFloats: number, len: number) => number;
  attentionDot64: (
    q: Float32Array,
    qOffset: number,
    k: Float32Array,
    kOffset: number
  ) => number;
}

let cachedWasmKernel: WasmCpuKernel | null = null;

export function getWasmCpuKernel(): WasmCpuKernel {
  if (cachedWasmKernel) return cachedWasmKernel;

  try {
    const mod = new WebAssembly.Module(WASM_DOT_F32_BYTES);
    const instance = new WebAssembly.Instance(mod);
    const memory = instance.exports.memory as WebAssembly.Memory;
    const rawDot = instance.exports.dot_f32 as (
      ptrA: number,
      ptrB: number,
      len: number
    ) => number;
    const floatArena = new Float32Array(memory.buffer);

    cachedWasmKernel = {
      isSupported: true,
      memory,
      floatArena,
      dotF32: (aOffsetFloats: number, bOffsetFloats: number, len: number) =>
        rawDot(aOffsetFloats << 2, bOffsetFloats << 2, len),
      attentionDot64: (
        q: Float32Array,
        qOffset: number,
        k: Float32Array,
        kOffset: number
      ) => {
        for (let i = 0; i < 64; i++) {
          floatArena[i] = q[qOffset + i];
          floatArena[64 + i] = k[kOffset + i];
        }
        return rawDot(0, 256, 64);
      },
    };
    return cachedWasmKernel;
  } catch {
    const fallbackMem = new WebAssembly.Memory({ initial: 1 });
    const floatArena = new Float32Array(fallbackMem.buffer);
    cachedWasmKernel = {
      isSupported: false,
      memory: fallbackMem,
      floatArena,
      dotF32: (aOffsetFloats: number, bOffsetFloats: number, len: number) => {
        let sum = 0;
        for (let i = 0; i < len; i++) {
          sum = Math.fround(
            sum + Math.fround(floatArena[aOffsetFloats + i] * floatArena[bOffsetFloats + i])
          );
        }
        return sum;
      },
      attentionDot64: (q, qOffset, k, kOffset) => {
        let sum = 0;
        for (let i = 0; i < 64; i++) {
          sum += q[qOffset + i] * k[kOffset + i];
        }
        return sum;
      },
    };
    return cachedWasmKernel;
  }
}
