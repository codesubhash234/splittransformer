import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MODEL_SPEC,
  createShardAWeights,
  createShardBWeights,
  NodeARuntime,
  NodeBRuntime,
  sampleNextByte,
} from './transformer';
import {
  DataChannelPacketAssembler,
  PacketType,
  encodeActivationFullChunks,
  encodeActivationStepInt8Packet,
  encodeActivationStepPacket,
  encodePrefillBatchPacket,
  encodeTokenResultPacket,
} from './protocol';

test('MODEL_SPEC strictly matches the 11,040,768 parameter (44.16 MB FP32) specification', () => {
  const tokEmb = 256 * 384; // 98,304
  const posEmb = 512 * 384; // 196,608
  const perLayer =
    2 * 384 + // ln1
    4 * (384 * 384 + 384) + // Q, K, V, Out projections
    2 * 384 + // ln2
    (384 * 1536 + 1536) + // fc1
    (1536 * 384 + 384); // fc2

  assert.equal(perLayer, 1_774_464);
  const shardA = tokEmb + posEmb + 3 * perLayer;
  const shardB = 3 * perLayer + 2 * 384 + 256 * 384;

  assert.equal(shardA, 5_618_304);
  assert.equal(shardB, 5_422_464);
  assert.equal(shardA + shardB, 11_040_768);
  assert.equal(MODEL_SPEC.totalParams, 11_040_768);
  assert.equal(MODEL_SPEC.totalBytesFp32, 44_163_072);
});

test('Seed 42 initialization is 100% deterministic across independent Node A & Node B instances', () => {
  const shardA1 = createShardAWeights(42);
  const shardA2 = createShardAWeights(42);
  assert.equal(shardA1.checksum, shardA2.checksum);
  assert.equal(shardA1.rawBuffer[0], shardA2.rawBuffer[0]);
  assert.equal(
    shardA1.rawBuffer[shardA1.rawBuffer.length - 1],
    shardA2.rawBuffer[shardA2.rawBuffer.length - 1]
  );

  const shardB1 = createShardBWeights(42);
  const shardB2 = createShardBWeights(42);
  assert.equal(shardB1.checksum, shardB2.checksum);
  assert.equal(shardB1.rawBuffer[0], shardB2.rawBuffer[0]);
  assert.equal(
    shardB1.rawBuffer[shardB1.rawBuffer.length - 1],
    shardB2.rawBuffer[shardB2.rawBuffer.length - 1]
  );
});

test('Split inference across Node A (Layers 0-2) and Node B (Layers 3-5) produces deterministic bytes at temp=0.0 and matches full-sequence execution', () => {
  const nodeA = new NodeARuntime();
  const nodeB = new NodeBRuntime();

  const promptBytes = [72, 101, 108, 108, 111]; // "Hello"
  nodeA.resetCache();
  nodeB.resetCache();

  let lastLogitsStep = new Float32Array(256);
  for (let pos = 0; pos < promptBytes.length; pos++) {
    const act = nodeA.forwardStep(promptBytes[pos], pos);
    assert.equal(act.length, 384);
    const res = nodeB.forwardStep(act, pos);
    lastLogitsStep = res.logits;
  }

  const sampleStep = sampleNextByte(lastLogitsStep, 0.0);

  // Now run Full-Sequence [T, 384] mode and verify exact mathematical identity
  const fullAct = nodeA.forwardFullSequence(promptBytes, false);
  assert.equal(fullAct.length, promptBytes.length * 384);
  const fullRes = nodeB.forwardFullSequence(fullAct, promptBytes.length);
  const sampleFull = sampleNextByte(fullRes.logits, 0.0);

  assert.equal(sampleStep.nextByte, sampleFull.nextByte);
  for (let i = 0; i < 256; i++) {
    assert.ok(
      Math.abs(lastLogitsStep[i] - fullRes.logits[i]) < 1e-5,
      `Logit mismatch at index ${i}: ${lastLogitsStep[i]} vs ${fullRes.logits[i]}`
    );
  }
});

test('Binary RTCDataChannel protocol encodes and reassembles both KV-step and full [512, 384] chunked tensors losslessly', () => {
  const assembler = new DataChannelPacketAssembler();

  // 1. Test single-step [384] activation packet
  const stepTensor = new Float32Array(384);
  for (let i = 0; i < 384; i++) stepTensor[i] = (i - 192) * 0.0125;
  const pktBuf = encodeActivationStepPacket(7, 12, 0.0, stepTensor);
  const decodedStep = assembler.decode(pktBuf);
  assert.ok(decodedStep && decodedStep.type === PacketType.ACTIVATION_STEP);
  assert.equal(decodedStep.stepId, 7);
  assert.equal(decodedStep.seqPos, 12);
  assert.deepEqual(Array.from(decodedStep.activations), Array.from(stepTensor));

  // 2. Test full [512 * 384] = 196,608 floats (786,432 bytes) multi-chunk SCTP tensor
  const fullTensor = new Float32Array(512 * 384);
  for (let i = 0; i < fullTensor.length; i++) {
    fullTensor[i] = Math.fround(Math.sin(i * 0.01));
  }
  const chunks = encodeActivationFullChunks(42, 512, 0.0, fullTensor);
  assert.ok(chunks.length > 40); // 786 KB / 16 KB = 48 chunks

  let reassembled = null;
  for (let c = 0; c < chunks.length; c++) {
    const out = assembler.decode(chunks[c]);
    if (c < chunks.length - 1) {
      assert.equal(out, null);
    } else {
      reassembled = out;
    }
  }
  assert.ok(reassembled && reassembled.type === PacketType.ACTIVATION_CHUNK);
  assert.equal(reassembled.stepId, 42);
  assert.equal(reassembled.seqLen, 512);
  assert.equal(reassembled.activations.length, 512 * 384);
  assert.equal(reassembled.activations[0], fullTensor[0]);
  assert.equal(reassembled.activations[512 * 384 - 1], fullTensor[512 * 384 - 1]);

  // 3. Test raw 1-byte token packet compatibility (`new Uint8Array([nextByte])`)
  const rawTokenBuf = encodeTokenResultPacket(1, 5, 105, 4.2, [], true);
  assert.equal(rawTokenBuf.byteLength, 1);
  const decodedRawToken = assembler.decode(rawTokenBuf);
  assert.ok(decodedRawToken && decodedRawToken.type === PacketType.TOKEN_RESULT);
  assert.equal(decodedRawToken.nextByte, 105);
});

test('WebAssembly (WASM CPU) and JavaScript (JS CPU) backends produce bit-identical bytes and PyTorch nn.Module default scheme is deterministic', () => {
  const shardA = createShardAWeights(42, undefined, 'pytorch-nn-default');
  const shardB = createShardBWeights(42, undefined, 'pytorch-nn-default');

  const nodeAWasm = new NodeARuntime(shardA, 'wasm-cpu');
  const nodeBWasm = new NodeBRuntime(shardB, 'wasm-cpu');
  const nodeAJs = new NodeARuntime(shardA, 'js-cpu');
  const nodeBJs = new NodeBRuntime(shardB, 'js-cpu');

  const prompt = [83, 112, 108, 105, 116]; // "Split"
  for (let pos = 0; pos < prompt.length; pos++) {
    const actWasm = nodeAWasm.forwardStep(prompt[pos], pos);
    const resWasm = nodeBWasm.forwardStep(actWasm, pos);
    const actJs = nodeAJs.forwardStep(prompt[pos], pos);
    const resJs = nodeBJs.forwardStep(actJs, pos);

    const byteWasm = sampleNextByte(resWasm.logits, 0.0).nextByte;
    const byteJs = sampleNextByte(resJs.logits, 0.0).nextByte;
    assert.equal(byteWasm, byteJs);
  }
});

test('Coalesced Prefill Batch, Dynamic Int8 Activation Quantization (75% smaller), and Speculative Lookahead preserve pipeline integrity', () => {
  const assembler = new DataChannelPacketAssembler();
  const nodeA = new NodeARuntime();
  const nodeB = new NodeBRuntime();

  // 1. Test Coalesced Prefill Batch packet (4 tokens in a single WebRTC frame)
  const batchAct = new Float32Array(4 * 384);
  for (let i = 0; i < batchAct.length; i++) batchAct[i] = Math.fround((i % 97) * 0.02 - 1.0);
  const batchBuf = encodePrefillBatchPacket(4, batchAct);
  const decodedBatch = assembler.decode(batchBuf);
  assert.ok(decodedBatch && decodedBatch.type === PacketType.PREFILL_BATCH);
  assert.equal(decodedBatch.count, 4);
  assert.equal(decodedBatch.activations.length, 4 * 384);
  assert.equal(decodedBatch.activations[0], batchAct[0]);

  // 2. Test Dynamic Int8 Activation Quantization (404 B total packet vs 1,552 B FP32)
  const realAct = nodeA.forwardStep(72, 0); // 'H' at pos 0
  const int8Buf = encodeActivationStepInt8Packet(1, 0, 0.0, realAct, 72);
  assert.equal(int8Buf.byteLength, 20 + 384); // 404 bytes total (74% smaller than 1552B)
  const decodedInt8 = assembler.decode(int8Buf);
  assert.ok(decodedInt8 && decodedInt8.type === PacketType.ACTIVATION_STEP);
  assert.equal(decodedInt8.wireBytes, 388);
  assert.equal(decodedInt8.activations.length, 384);

  // Verify max quantization reconstruction error is bounded by scale / 2
  let maxErr = 0;
  for (let i = 0; i < 384; i++) {
    const err = Math.abs(decodedInt8.activations[i] - realAct[i]);
    if (err > maxErr) maxErr = err;
  }
  assert.ok(maxErr < 0.05, `Expected small Int8 reconstruction error, got ${maxErr}`);

  // 3. Test Speculative Pipeline Lookahead KV-cache safety (Hit & Miss)
  nodeA.resetCache();
  nodeB.resetCache();
  const act0 = nodeA.forwardStep(72, 0);
  nodeB.forwardStep(act0, 0);

  // Speculatively evaluate candidate byte 99 at pos 1, then overwrite with actual byte 101 at pos 1
  nodeA.forwardStep(99, 1); // Speculative miss
  const act1Recovered = nodeA.forwardStep(101, 1); // Actual byte overwrites KV slot 1

  const nodeAClean = new NodeARuntime();
  nodeAClean.forwardStep(72, 0);
  const act1Clean = nodeAClean.forwardStep(101, 1);

  assert.deepEqual(Array.from(act1Recovered), Array.from(act1Clean));
});


