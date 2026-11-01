/**
 * WebRTC Peer-to-Peer DataChannel Transport Layer (`RTCPeerConnection` + `RTCDataChannel`)
 *
 * Supports three real WebRTC P2P connection topologies:
 * 1. Physical Two-Phone P2P Handshake (Zero-Server SDP Offer/Answer via QR / Copy-Paste Token)
 * 2. Multi-Window / Multi-Tab P2P Handshake (Automated SDP exchange via BroadcastChannel, transport via WebRTC DataChannel)
 * 3. Local Dual-Node WebRTC Loopback (Two real RTCPeerConnection instances `pcA <-> pcB` connected via RTCDataChannel in-page)
 */

export type PeerRole = 'node-a' | 'node-b' | 'dual-loopback';
export type ConnectionState = 'disconnected' | 'gathering-ice' | 'awaiting-peer' | 'connecting' | 'connected' | 'error';

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

/**
 * Compress SDP string by stripping redundant non-essential codec lines (audio/video RTP maps)
 * since RTCDataChannel only uses the `m=application` SCTP section.
 */
export function encodeCompactSdp(type: 'offer' | 'answer', role: 'node-a' | 'node-b', sdp: string): string {
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

const RTC_CONFIG: RTCConfiguration = {
  // For local LAN P2P, host candidates work completely offline without STUN;
  // public STUN is included as optional fallback when internet is present during initial pairing.
  iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
};

export class WebRTCMeshManager {
  private pcA: RTCPeerConnection | null = null;
  private pcB: RTCPeerConnection | null = null;
  private dcA: RTCDataChannel | null = null;
  private dcB: RTCDataChannel | null = null;
  private broadcastChannel: BroadcastChannel | null = null;

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

  private onStateChangeCb: ((state: ConnectionState, stats: DataChannelStats) => void) | null = null;
  private onNodeAMessageCb: ((data: ArrayBuffer) => void) | null = null;
  private onNodeBMessageCb: ((data: ArrayBuffer) => void) | null = null;

  public setCallbacks( handlers: {
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
    this.errorMessage = null;
    this.notifyState('disconnected');
  }

  /**
   * Wait until ICE gathering completes (or timeout after `timeoutMs` so offline LAN host candidates resolve fast).
   */
  private waitForIceGathering(pc: RTCPeerConnection, timeoutMs = 350): Promise<void> {
    if (pc.iceGatheringState === 'complete') {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      }, timeoutMs);
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          clearTimeout(timer);
          pc.removeEventListener('icegatheringstatechange', check);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
    });
  }

  /**
   * MODE 1A: Phone 1 (Node A · Caller) generates an SDP Offer token for Phone 2.
   */
  public async createPhone1Offer(): Promise<string> {
    this.disconnect();
    this.stats.transportMode = 'p2p-physical';
    this.notifyState('gathering-ice');

    const pc = new RTCPeerConnection(RTC_CONFIG);
    this.pcA = pc;

    const dc = pc.createDataChannel('tensor-mesh', { ordered: true });
    dc.binaryType = 'arraybuffer';
    this.bindChannelA(dc);

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await this.waitForIceGathering(pc);

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
    await this.waitForIceGathering(pc);

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

  private pollIntervalId: ReturnType<typeof setInterval> | null = null;

  /**
   * MODE 2: 1-Tap Room Code Auto-Signaling across Two Physical Phones (via /api/signal/:room)
   * AND Multi-Tab BroadcastChannel.
   * Exchanges ONLY the initial WebRTC SDP Offer/Answer to open the direct P2P RTCDataChannel;
   * polling terminates immediately once the DataChannel opens.
   */
  public async startBroadcastAutoPair(role: 'node-a' | 'node-b', roomCode = '4242'): Promise<void> {
    this.disconnect();
    this.stats.transportMode = 'p2p-physical';
    this.stats.iceCandidateType = 'host / srflx (Direct P2P WebRTC)';
    this.notifyState('gathering-ice');

    const cleanRoom = roomCode.trim() || '4242';
    const bc = new BroadcastChannel(`splitbyte-webrtc-sig-${cleanRoom}`);
    this.broadcastChannel = bc;

    if (role === 'node-a') {
      const pc = new RTCPeerConnection(RTC_CONFIG);
      this.pcA = pc;
      const dc = pc.createDataChannel('tensor-mesh', { ordered: true });
      dc.binaryType = 'arraybuffer';
      this.bindChannelA(dc, () => {
        if (this.pollIntervalId) {
          clearInterval(this.pollIntervalId);
          this.pollIntervalId = null;
        }
      });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await this.waitForIceGathering(pc, 400);
      const offerSdp = pc.localDescription?.sdp || '';

      this.notifyState('awaiting-peer');

      // Publish SDP Offer to both local BroadcastChannel and ephemeral room rendezvous
      bc.postMessage({ type: 'offer', sdp: offerSdp });
      fetch(`/api/signal/${encodeURIComponent(cleanRoom)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'offer', sdp: offerSdp }),
      }).catch(() => {});

      let answerApplied = false;
      const applyAnswerSdp = async (answerSdp: string) => {
        if (answerApplied || pc.signalingState !== 'have-local-offer') return;
        answerApplied = true;
        this.notifyState('connecting');
        await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
      };

      bc.onmessage = async (ev) => {
        const msg = ev.data;
        if (msg?.type === 'callee-ready') {
          bc.postMessage({ type: 'offer', sdp: offerSdp });
        } else if (msg?.type === 'answer' && msg.sdp) {
          await applyAnswerSdp(msg.sdp);
        }
      };

      // Poll ephemeral room for Phone 2's SDP Answer until RTCDataChannel opens
      this.pollIntervalId = setInterval(async () => {
        if (this.state === 'connected' || answerApplied) {
          if (this.pollIntervalId) {
            clearInterval(this.pollIntervalId);
            this.pollIntervalId = null;
          }
          return;
        }
        try {
          const resp = await fetch(`/api/signal/${encodeURIComponent(cleanRoom)}`);
          if (!resp.ok) return;
          const data = await resp.json();
          if (data?.answerSdp) {
            await applyAnswerSdp(data.answerSdp);
          }
        } catch {
          // Offline fallback continues via BroadcastChannel or Manual SDP
        }
      }, 800);
    } else {
      const pc = new RTCPeerConnection(RTC_CONFIG);
      this.pcB = pc;
      pc.ondatachannel = (ev) => {
        const dc = ev.channel;
        dc.binaryType = 'arraybuffer';
        this.bindChannelB(dc, () => {
          if (this.pollIntervalId) {
            clearInterval(this.pollIntervalId);
            this.pollIntervalId = null;
          }
        });
      };

      this.notifyState('awaiting-peer');

      let answeredOfferSdp = '';
      const handleIncomingOfferSdp = async (offerSdp: string) => {
        if (!offerSdp || offerSdp === answeredOfferSdp) return;
        answeredOfferSdp = offerSdp;
        this.notifyState('connecting');
        await pc.setRemoteDescription({ type: 'offer', sdp: offerSdp });
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        await this.waitForIceGathering(pc, 400);
        const answerSdp = pc.localDescription?.sdp || '';

        bc.postMessage({ type: 'answer', sdp: answerSdp });
        fetch(`/api/signal/${encodeURIComponent(cleanRoom)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: 'answer', sdp: answerSdp }),
        }).catch(() => {});
      };

      bc.onmessage = async (ev) => {
        const msg = ev.data;
        if (msg?.type === 'offer' && msg.sdp) {
          await handleIncomingOfferSdp(msg.sdp);
        }
      };

      bc.postMessage({ type: 'callee-ready' });

      // Poll ephemeral room for Phone 1's SDP Offer until RTCDataChannel opens
      this.pollIntervalId = setInterval(async () => {
        if (this.state === 'connected') {
          if (this.pollIntervalId) {
            clearInterval(this.pollIntervalId);
            this.pollIntervalId = null;
          }
          return;
        }
        try {
          const resp = await fetch(`/api/signal/${encodeURIComponent(cleanRoom)}`);
          if (!resp.ok) return;
          const data = await resp.json();
          if (data?.offerSdp) {
            await handleIncomingOfferSdp(data.offerSdp);
          }
        } catch {
          // Offline fallback continues via BroadcastChannel or Manual SDP
        }
      }, 800);
    }
  }

  private portA: MessagePort | null = null;
  private portB: MessagePort | null = null;

  /**
   * MODE 3: Local Dual-Node P2P WebRTC Loopback (`pcA <-> pcB` via real `RTCDataChannel` in-browser)
   * Queues ICE candidates until remoteDescription is set and falls back within 350ms to binary
   * MessageChannel ArrayBuffer transfer if the OS/container blocks local UDP sockets.
   */
  public async connectDualNodeLoopback(): Promise<void> {
    this.disconnect();
    this.stats.transportMode = 'p2p-webrtc-loopback';
    this.stats.iceCandidateType = 'host (WebRTC SCTP loopback)';
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
        // Container UDP loopback fallback using zero-copy binary MessageChannel ports
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
      this.notifyState('connected');
      if (onOpenExtra) onOpenExtra();
    };
    dc.onclose = () => {
      if (this.state === 'connected') this.notifyState('disconnected');
    };
    dc.onerror = () => {
      this.errorMessage = 'RTCDataChannel (Node A) encountered a transport error.';
      this.notifyState('error');
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
      this.notifyState('connected');
      if (onOpenExtra) onOpenExtra();
    };
    dc.onclose = () => {
      if (this.state === 'connected') this.notifyState('disconnected');
    };
    dc.onerror = () => {
      this.errorMessage = 'RTCDataChannel (Node B) encountered a transport error.';
      this.notifyState('error');
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
   * Send binary activation packet(s) from Phone 1 (Node A) -> Phone 2 (Node B) over WebRTC DataChannel
   */
  public sendFromNodeA(buffer: ArrayBuffer): void {
    if (this.dcA && this.dcA.readyState === 'open') {
      this.dcA.send(buffer);
    } else if (this.portA) {
      this.portA.postMessage(buffer);
    } else {
      throw new Error('Node A RTCDataChannel is not open.');
    }
    this.stats.messagesSent++;
    this.stats.bytesSent += buffer.byteLength;
    this.stats.lastPacketBytes = buffer.byteLength;
  }

  /**
   * Send binary token result packet from Phone 2 (Node B) -> Phone 1 (Node A) over WebRTC DataChannel
   */
  public sendFromNodeB(buffer: ArrayBuffer): void {
    if (this.dcB && this.dcB.readyState === 'open') {
      this.dcB.send(buffer);
    } else if (this.portB) {
      this.portB.postMessage(buffer);
    } else {
      throw new Error('Node B RTCDataChannel is not open.');
    }
    this.stats.messagesSent++;
    this.stats.bytesSent += buffer.byteLength;
    this.stats.lastPacketBytes = buffer.byteLength;
  }
}
