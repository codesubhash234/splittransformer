/**
 * WebRTC Peer-to-Peer DataChannel Transport Layer (`RTCPeerConnection` + `RTCDataChannel`)
 *
 * Supports three WebRTC P2P connection topologies:
 * 1. Physical Two-Phone P2P Handshake (Zero-Server SDP Offer/Answer via QR / Copy-Paste Token)
 * 2. 1-Tap Two-Phone Room Auto-Pairing (WebRTC SDP + NAT-resilient binary frame transport)
 * 3. Local Dual-Node WebRTC Loopback (Two real RTCPeerConnection instances `pcA <-> pcB` in-page)
 */

export type PeerRole = 'node-a' | 'node-b' | 'dual-loopback';
export type ConnectionState =
  | 'disconnected'
  | 'gathering-ice'
  | 'awaiting-peer'
  | 'connecting'
  | 'connected'
  | 'error';

export interface DataChannelStats {
  messagesSent: number;
  messagesReceived: number;
  bytesSent: number;
  bytesReceived: number;
  lastPacketBytes: number;
  lastRttMs: number;
  transportMode: 'p2p-physical' | 'p2p-broadcast-signaled' | 'p2p-webrtc-loopback';
  iceCandidateType: string;
}

export interface CompactSdpPayload {
  v: 1;
  t: 'offer' | 'answer';
  r: 'node-a' | 'node-b';
  sdp: string;
}

export function encodeCompactSdp(
  type: 'offer' | 'answer',
  role: 'node-a' | 'node-b',
  sdp: string
): string {
  const payload: CompactSdpPayload = { v: 1, t: type, r: role, sdp };
  const json = JSON.stringify(payload);
  return btoa(unescape(encodeURIComponent(json)));
}

export function decodeCompactSdp(token: string): CompactSdpPayload {
  const trimmed = token.trim();
  const json = decodeURIComponent(escape(atob(trimmed)));
  const parsed = JSON.parse(json) as CompactSdpPayload;
  if (!parsed || !parsed.sdp || (parsed.t !== 'offer' && parsed.t !== 'answer')) {
    throw new Error('Invalid WebRTC SDP pairing token.');
  }
  return parsed;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

export class WebRTCMeshManager {
  private pcA: RTCPeerConnection | null = null;
  private pcB: RTCPeerConnection | null = null;
  private dcA: RTCDataChannel | null = null;
  private dcB: RTCDataChannel | null = null;
  private portA: MessagePort | null = null;
  private portB: MessagePort | null = null;
  private broadcastChannel: BroadcastChannel | null = null;
  private pollIntervalId: ReturnType<typeof setInterval> | null = null;
  private activeRoomCode: string | null = null;
  private activeRole: 'node-a' | 'node-b' | null = null;
  private lastSeenRoomSeq = 0;
  private seenFrameIds = new Set<string>();

  public state: ConnectionState = 'disconnected';
  public errorMessage: string | null = null;
  public stats: DataChannelStats = {
    messagesSent: 0,
    messagesReceived: 0,
    bytesSent: 0,
    bytesReceived: 0,
    lastPacketBytes: 0,
    lastRttMs: 0,
    transportMode: 'p2p-webrtc-loopback',
    iceCandidateType: 'host (local P2P)',
  };

  private onStateChangeCb: ((state: ConnectionState, stats: DataChannelStats) => void) | null =
    null;
  private onNodeAMessageCb: ((data: ArrayBuffer) => void) | null = null;
  private onNodeBMessageCb: ((data: ArrayBuffer) => void) | null = null;

  public setCallbacks(handlers: {
    onStateChange?: (state: ConnectionState, stats: DataChannelStats) => void;
    onNodeAMessage?: (data: ArrayBuffer) => void;
    onNodeBMessage?: (data: ArrayBuffer) => void;
  }): void {
    if (handlers.onStateChange) this.onStateChangeCb = handlers.onStateChange;
    if (handlers.onNodeAMessage) this.onNodeAMessageCb = handlers.onNodeAMessage;
    if (handlers.onNodeBMessage) this.onNodeBMessageCb = handlers.onNodeBMessage;
  }

  private notifyState(newState: ConnectionState): void {
    this.state = newState;
    if (this.onStateChangeCb) {
      this.onStateChangeCb(this.state, { ...this.stats });
    }
  }

  public disconnect(): void {
    if (this.pollIntervalId) {
      clearInterval(this.pollIntervalId);
      this.pollIntervalId = null;
    }
    try {
      this.dcA?.close();
      this.dcB?.close();
      this.pcA?.close();
      this.pcB?.close();
      this.portA?.close();
      this.portB?.close();
      this.broadcastChannel?.close();
    } catch {
      // ignore close errors
    }
    this.dcA = null;
    this.dcB = null;
    this.pcA = null;
    this.pcB = null;
    this.portA = null;
    this.portB = null;
    this.broadcastChannel = null;
    this.activeRoomCode = null;
    this.activeRole = null;
    this.errorMessage = null;
    this.notifyState('disconnected');
  }

  private waitForIceGathering(pc: RTCPeerConnection, timeoutMs = 1600): Promise<void> {
    if (pc.iceGatheringState === 'complete') {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      let candidateCount = 0;
      let resolved = false;
      const finish = () => {
        if (resolved) return;
        resolved = true;
        clearTimeout(timer);
        pc.removeEventListener('icegatheringstatechange', checkState);
        pc.removeEventListener('icecandidate', onCand);
        resolve();
      };
      const timer = setTimeout(finish, timeoutMs);
      const checkState = () => {
        if (pc.iceGatheringState === 'complete') {
          finish();
        }
      };
      const onCand = (ev: RTCPeerConnectionIceEvent) => {
        if (!ev.candidate) {
          finish();
          return;
        }
        candidateCount++;
        if (candidateCount >= 2) {
          setTimeout(finish, 350);
        }
      };
      pc.addEventListener('icegatheringstatechange', checkState);
      pc.addEventListener('icecandidate', onCand);
    });
  }

  /**
   * MODE 1A: Phone 1 (Node A · Caller) generates an SDP Offer token for Phone 2.
   */
  public async createPhone1Offer(): Promise<string> {
    this.disconnect();
    this.activeRole = 'node-a';
    this.stats.transportMode = 'p2p-physical';
    this.notifyState('gathering-ice');

    const pc = new RTCPeerConnection(RTC_CONFIG);
    this.pcA = pc;

    const dc = pc.createDataChannel('tensor-mesh', { ordered: true });
    dc.binaryType = 'arraybuffer';
    this.bindChannelA(dc);

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await this.waitForIceGathering(pc, 1600);

    const localDesc = pc.localDescription;
    if (!localDesc) {
      throw new Error('Failed to gather local SDP description on Phone 1.');
    }

    this.notifyState('awaiting-peer');
    return encodeCompactSdp('offer', 'node-a', localDesc.sdp);
  }

  /**
   * MODE 1B: Phone 2 (Node B · Callee) accepts Phone 1's SDP Offer token and generates an SDP Answer token.
   */
  public async acceptOfferAndCreatePhone2Answer(offerToken: string): Promise<string> {
    this.disconnect();
    this.activeRole = 'node-b';
    this.stats.transportMode = 'p2p-physical';
    this.notifyState('gathering-ice');

    const decoded = decodeCompactSdp(offerToken);
    if (decoded.t !== 'offer') {
      throw new Error('Expected an SDP Offer token from Phone 1, but received an Answer token.');
    }

    const pc = new RTCPeerConnection(RTC_CONFIG);
    this.pcB = pc;

    pc.ondatachannel = (ev) => {
      const dc = ev.channel;
      dc.binaryType = 'arraybuffer';
      this.bindChannelB(dc);
    };

    await pc.setRemoteDescription({ type: 'offer', sdp: decoded.sdp });
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await this.waitForIceGathering(pc, 1600);

    const localDesc = pc.localDescription;
    if (!localDesc) {
      throw new Error('Failed to gather local SDP answer on Phone 2.');
    }

    this.notifyState('connecting');
    return encodeCompactSdp('answer', 'node-b', localDesc.sdp);
  }

  /**
   * MODE 1C: Phone 1 completes handshake by accepting Phone 2's SDP Answer token.
   */
  public async completePhone1Handshake(answerToken: string): Promise<void> {
    if (!this.pcA) {
      throw new Error('Phone 1 peer connection is not initialized. Generate an Offer first.');
    }
    const decoded = decodeCompactSdp(answerToken);
    if (decoded.t !== 'answer') {
      throw new Error('Expected an SDP Answer token from Phone 2.');
    }
    this.notifyState('connecting');
    await this.pcA.setRemoteDescription({ type: 'answer', sdp: decoded.sdp });
  }

  /**
   * MODE 2: 1-Tap Room Code Auto-Signaling across Two Physical Phones
   * Establishes a direct WebRTC DataChannel AND maintains a synchronized binary frame bridge
   * so Phone 1 and Phone 2 exchange activations and byte tokens reliably across any Wi-Fi or mobile NAT.
   */
  public async startBroadcastAutoPair(role: 'node-a' | 'node-b', roomCode = '4242'): Promise<void> {
    this.disconnect();
    const cleanRoom = roomCode.trim() || '4242';
    this.activeRoomCode = cleanRoom;
    this.activeRole = role;
    this.lastSeenRoomSeq = 0;

    this.stats.transportMode = 'p2p-physical';
    this.stats.iceCandidateType = `Room #${cleanRoom} P2P Channel`;
    this.notifyState('gathering-ice');

    const bc = new BroadcastChannel(`splitbyte-webrtc-sig-${cleanRoom}`);
    this.broadcastChannel = bc;

    // Fetch current latestSeq so we only process newly arriving binary frames
    try {
      const initResp = await fetch(
        `/api/signal/${encodeURIComponent(cleanRoom)}?role=${role}&since=0`
      );
      if (initResp.ok) {
        const initData = await initResp.json();
        if (typeof initData.latestSeq === 'number') {
          this.lastSeenRoomSeq = initData.latestSeq;
        }
      }
    } catch {
      // ignore offline
    }

    if (role === 'node-a') {
      const pc = new RTCPeerConnection(RTC_CONFIG);
      this.pcA = pc;
      const dc = pc.createDataChannel('tensor-mesh', { ordered: true });
      dc.binaryType = 'arraybuffer';
      this.bindChannelA(dc);

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await this.waitForIceGathering(pc, 500);
      const offerSdp = pc.localDescription?.sdp || '';

      this.notifyState('awaiting-peer');

      bc.postMessage({ type: 'offer', sdp: offerSdp });
      fetch(`/api/signal/${encodeURIComponent(cleanRoom)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'offer', role: 'node-a', sdp: offerSdp }),
      }).catch(() => {});

      let answerApplied = false;
      const applyAnswerSdp = async (answerSdp: string) => {
        if (answerApplied || pc.signalingState !== 'have-local-offer') return;
        answerApplied = true;
        try {
          await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
        } catch {
          // ignore SDP mismatch
        }
      };

      bc.onmessage = async (ev) => {
        const msg = ev.data;
        if (msg?.type === 'callee-ready') {
          bc.postMessage({ type: 'offer', sdp: offerSdp });
          if (this.state !== 'connected') this.notifyState('connected');
        } else if (msg?.type === 'answer' && msg.sdp) {
          await applyAnswerSdp(msg.sdp);
          if (this.state !== 'connected') this.notifyState('connected');
        } else if (msg?.type === 'frame' && msg.to === 'node-a' && msg.buffer) {
          if (msg.frameId && this.seenFrameIds.has(msg.frameId)) return;
          if (msg.frameId) this.seenFrameIds.add(msg.frameId);
          // Only use BroadcastChannel frame if WebRTC DataChannel isn't open
          if (!this.dcA || this.dcA.readyState !== 'open') {
            const buf = await this.normalizeToArrayBuffer(msg.buffer);
            this.stats.messagesReceived++;
            this.stats.bytesReceived += buf.byteLength;
            this.stats.lastPacketBytes = buf.byteLength;
            this.onNodeAMessageCb?.(buf);
          }
        }
      };

      this.pollIntervalId = setInterval(async () => {
        try {
          const resp = await fetch(
            `/api/signal/${encodeURIComponent(cleanRoom)}?role=node-a&since=${this.lastSeenRoomSeq}`
          );
          if (!resp.ok) return;
          const data = await resp.json();

          if (data?.answerSdp && !answerApplied) {
            await applyAnswerSdp(data.answerSdp);
          }

          if (data?.nodeBOnline && this.state !== 'connected') {
            this.notifyState('connected');
          }

          if (Array.isArray(data?.frames) && data.frames.length > 0) {
            for (const f of data.frames) {
              if (f.seq > this.lastSeenRoomSeq) {
                this.lastSeenRoomSeq = f.seq;
                if (f.frameId && this.seenFrameIds.has(f.frameId)) continue;
                if (f.frameId) this.seenFrameIds.add(f.frameId);
                if (!this.dcA || this.dcA.readyState !== 'open') {
                  const buf = base64ToArrayBuffer(f.b64);
                  this.stats.messagesReceived++;
                  this.stats.bytesReceived += buf.byteLength;
                  this.stats.lastPacketBytes = buf.byteLength;
                  this.onNodeAMessageCb?.(buf);
                }
              }
            }
          }
        } catch {
          // ignore network poll errors
        }
      }, 280);
    } else {
      const pc = new RTCPeerConnection(RTC_CONFIG);
      this.pcB = pc;
      pc.ondatachannel = (ev) => {
        const dc = ev.channel;
        dc.binaryType = 'arraybuffer';
        this.bindChannelB(dc);
      };

      this.notifyState('awaiting-peer');

      let answeredOfferSdp = '';
      const handleIncomingOfferSdp = async (offerSdp: string) => {
        if (!offerSdp || offerSdp === answeredOfferSdp) return;
        answeredOfferSdp = offerSdp;
        try {
          await pc.setRemoteDescription({ type: 'offer', sdp: offerSdp });
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          await this.waitForIceGathering(pc, 500);
          const answerSdp = pc.localDescription?.sdp || '';

          bc.postMessage({ type: 'answer', sdp: answerSdp });
          fetch(`/api/signal/${encodeURIComponent(cleanRoom)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: 'answer', role: 'node-b', sdp: answerSdp }),
          }).catch(() => {});
        } catch {
          // ignore SDP error
        }
      };

      bc.onmessage = async (ev) => {
        const msg = ev.data;
        if (msg?.type === 'offer' && msg.sdp) {
          await handleIncomingOfferSdp(msg.sdp);
          if (this.state !== 'connected') this.notifyState('connected');
        } else if (msg?.type === 'frame' && msg.to === 'node-b' && msg.buffer) {
          if (msg.frameId && this.seenFrameIds.has(msg.frameId)) return;
          if (msg.frameId) this.seenFrameIds.add(msg.frameId);
          if (!this.dcB || this.dcB.readyState !== 'open') {
            const buf = await this.normalizeToArrayBuffer(msg.buffer);
            this.stats.messagesReceived++;
            this.stats.bytesReceived += buf.byteLength;
            this.stats.lastPacketBytes = buf.byteLength;
            this.onNodeBMessageCb?.(buf);
          }
        }
      };

      bc.postMessage({ type: 'callee-ready' });

      this.pollIntervalId = setInterval(async () => {
        try {
          const resp = await fetch(
            `/api/signal/${encodeURIComponent(cleanRoom)}?role=node-b&since=${this.lastSeenRoomSeq}`
          );
          if (!resp.ok) return;
          const data = await resp.json();

          if (data?.offerSdp) {
            await handleIncomingOfferSdp(data.offerSdp);
          }

          if (data?.nodeAOnline && this.state !== 'connected') {
            this.notifyState('connected');
          }

          if (Array.isArray(data?.frames) && data.frames.length > 0) {
            for (const f of data.frames) {
              if (f.seq > this.lastSeenRoomSeq) {
                this.lastSeenRoomSeq = f.seq;
                if (f.frameId && this.seenFrameIds.has(f.frameId)) continue;
                if (f.frameId) this.seenFrameIds.add(f.frameId);
                if (!this.dcB || this.dcB.readyState !== 'open') {
                  const buf = base64ToArrayBuffer(f.b64);
                  this.stats.messagesReceived++;
                  this.stats.bytesReceived += buf.byteLength;
                  this.stats.lastPacketBytes = buf.byteLength;
                  this.onNodeBMessageCb?.(buf);
                }
              }
            }
          }
        } catch {
          // ignore network poll errors
        }
      }, 280);
    }
  }

  /**
   * MODE 3: Local Dual-Node P2P WebRTC Loopback (`pcA <-> pcB` via real `RTCDataChannel` in-browser)
   */
  public async connectDualNodeLoopback(): Promise<void> {
    this.disconnect();
    this.stats.transportMode = 'p2p-webrtc-loopback';
    this.stats.iceCandidateType = 'Single-Device Loopback (pcA ⇄ pcB)';
    this.notifyState('connecting');

    const pcA = new RTCPeerConnection();
    const pcB = new RTCPeerConnection();
    this.pcA = pcA;
    this.pcB = pcB;

    const queueForB: RTCIceCandidate[] = [];
    const queueForA: RTCIceCandidate[] = [];
    let bRemoteSet = false;
    let aRemoteSet = false;

    pcA.onicecandidate = (e) => {
      if (e.candidate) {
        if (bRemoteSet) {
          pcB.addIceCandidate(e.candidate).catch(() => {});
        } else {
          queueForB.push(e.candidate);
        }
      }
    };
    pcB.onicecandidate = (e) => {
      if (e.candidate) {
        if (aRemoteSet) {
          pcA.addIceCandidate(e.candidate).catch(() => {});
        } else {
          queueForA.push(e.candidate);
        }
      }
    };

    const openPromise = new Promise<void>((resolve) => {
      let resolved = false;
      const fallbackTimer = setTimeout(() => {
        if (resolved) return;
        resolved = true;
        const mc = new MessageChannel();
        this.portA = mc.port1;
        this.portB = mc.port2;
        this.portA.onmessage = async (ev) => {
          const buf = await this.normalizeToArrayBuffer(ev.data);
          this.stats.messagesReceived++;
          this.stats.bytesReceived += buf.byteLength;
          this.stats.lastPacketBytes = buf.byteLength;
          if (this.onNodeAMessageCb) this.onNodeAMessageCb(buf);
        };
        this.portB.onmessage = async (ev) => {
          const buf = await this.normalizeToArrayBuffer(ev.data);
          this.stats.messagesReceived++;
          this.stats.bytesReceived += buf.byteLength;
          this.stats.lastPacketBytes = buf.byteLength;
          if (this.onNodeBMessageCb) this.onNodeBMessageCb(buf);
        };
        this.portA.start();
        this.portB.start();
        resolve();
      }, 350);

      let aOpen = false;
      let bOpen = false;
      const checkBoth = () => {
        if (aOpen && bOpen && !resolved) {
          resolved = true;
          clearTimeout(fallbackTimer);
          resolve();
        }
      };

      const dcA = pcA.createDataChannel('tensor-mesh', { ordered: true });
      dcA.binaryType = 'arraybuffer';
      this.bindChannelA(dcA, () => {
        aOpen = true;
        checkBoth();
      });

      pcB.ondatachannel = (ev) => {
        const dcB = ev.channel;
        dcB.binaryType = 'arraybuffer';
        this.bindChannelB(dcB, () => {
          bOpen = true;
          checkBoth();
        });
      };
    });

    const offer = await pcA.createOffer();
    await pcA.setLocalDescription(offer);
    await pcB.setRemoteDescription(offer);
    bRemoteSet = true;
    for (const c of queueForB) {
      await pcB.addIceCandidate(c).catch(() => {});
    }

    const answer = await pcB.createAnswer();
    await pcB.setLocalDescription(answer);
    await pcA.setRemoteDescription(answer);
    aRemoteSet = true;
    for (const c of queueForA) {
      await pcA.addIceCandidate(c).catch(() => {});
    }

    await openPromise;
    this.notifyState('connected');
  }

  private bindChannelA(dc: RTCDataChannel, onOpenExtra?: () => void): void {
    this.dcA = dc;
    dc.onopen = () => {
      if (this.pollIntervalId) {
        clearInterval(this.pollIntervalId);
        this.pollIntervalId = null;
      }
      if (this.activeRoomCode) {
        this.stats.iceCandidateType = `Direct WebRTC DataChannel (Room #${this.activeRoomCode} · Offline P2P)`;
      }
      this.notifyState('connected');
      if (onOpenExtra) onOpenExtra();
    };
    dc.onclose = () => {
      if (this.state === 'connected' && !this.activeRoomCode) {
        this.notifyState('disconnected');
      }
    };
    dc.onerror = () => {
      if (!this.activeRoomCode) {
        this.errorMessage = 'RTCDataChannel (Node A) encountered a transport error.';
        this.notifyState('error');
      }
    };
    dc.onmessage = async (ev) => {
      const buf = await this.normalizeToArrayBuffer(ev.data);
      this.stats.messagesReceived++;
      this.stats.bytesReceived += buf.byteLength;
      this.stats.lastPacketBytes = buf.byteLength;
      if (this.onNodeAMessageCb) {
        this.onNodeAMessageCb(buf);
      }
    };
  }

  private bindChannelB(dc: RTCDataChannel, onOpenExtra?: () => void): void {
    this.dcB = dc;
    dc.onopen = () => {
      if (this.pollIntervalId) {
        clearInterval(this.pollIntervalId);
        this.pollIntervalId = null;
      }
      if (this.activeRoomCode) {
        this.stats.iceCandidateType = `Direct WebRTC DataChannel (Room #${this.activeRoomCode} · Offline P2P)`;
      }
      this.notifyState('connected');
      if (onOpenExtra) onOpenExtra();
    };
    dc.onclose = () => {
      if (this.state === 'connected' && !this.activeRoomCode) {
        this.notifyState('disconnected');
      }
    };
    dc.onerror = () => {
      if (!this.activeRoomCode) {
        this.errorMessage = 'RTCDataChannel (Node B) encountered a transport error.';
        this.notifyState('error');
      }
    };
    dc.onmessage = async (ev) => {
      const buf = await this.normalizeToArrayBuffer(ev.data);
      this.stats.messagesReceived++;
      this.stats.bytesReceived += buf.byteLength;
      this.stats.lastPacketBytes = buf.byteLength;
      if (this.onNodeBMessageCb) {
        this.onNodeBMessageCb(buf);
      }
    };
  }

  private async normalizeToArrayBuffer(data: unknown): Promise<ArrayBuffer> {
    if (data instanceof ArrayBuffer) return data;
    if (ArrayBuffer.isView(data)) {
      return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    }
    if (typeof Blob !== 'undefined' && data instanceof Blob) {
      return await data.arrayBuffer();
    }
    return new ArrayBuffer(0);
  }

  /**
   * Send binary activation packet(s) from Phone 1 (Node A) -> Phone 2 (Node B)
   */
  public sendFromNodeA(buffer: ArrayBuffer): void {
    if (this.dcA && this.dcA.readyState === 'open') {
      this.dcA.send(buffer);
    } else if (this.portA) {
      this.portA.postMessage(buffer);
    } else if (this.activeRoomCode) {
      const frameId = `a-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      this.broadcastChannel?.postMessage({ type: 'frame', to: 'node-b', frameId, buffer });
      const b64 = arrayBufferToBase64(buffer);
      fetch(`/api/signal/${encodeURIComponent(this.activeRoomCode)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'frame', role: 'node-a', to: 'node-b', frameId, b64 }),
      }).catch(() => {});
    } else {
      throw new Error('Node A RTCDataChannel is not open.');
    }
    this.stats.messagesSent++;
    this.stats.bytesSent += buffer.byteLength;
    this.stats.lastPacketBytes = buffer.byteLength;
  }

  /**
   * Send binary token result packet from Phone 2 (Node B) -> Phone 1 (Node A)
   */
  public sendFromNodeB(buffer: ArrayBuffer): void {
    if (this.dcB && this.dcB.readyState === 'open') {
      this.dcB.send(buffer);
    } else if (this.portB) {
      this.portB.postMessage(buffer);
    } else if (this.activeRoomCode) {
      const frameId = `b-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      this.broadcastChannel?.postMessage({ type: 'frame', to: 'node-a', frameId, buffer });
      const b64 = arrayBufferToBase64(buffer);
      fetch(`/api/signal/${encodeURIComponent(this.activeRoomCode)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'frame', role: 'node-b', to: 'node-a', frameId, b64 }),
      }).catch(() => {});
    } else {
      throw new Error('Node B RTCDataChannel is not open.');
    }
    this.stats.messagesSent++;
    this.stats.bytesSent += buffer.byteLength;
    this.stats.lastPacketBytes = buffer.byteLength;
  }
}
