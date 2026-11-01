import React, { useState } from 'react';
import { PeerRole, ConnectionState, DataChannelStats } from '../engine/webrtc';
import { generatePairingMatrix } from '../engine/pytorchScript';

interface PairingPanelProps {
  role: PeerRole;
  connectionState: ConnectionState;
  stats: DataChannelStats;
  errorMessage: string | null;
  onSelectRole: (role: PeerRole) => void;
  onConnectLoopback: () => Promise<void>;
  onStartBroadcastAutoPair: (role: 'node-a' | 'node-b', roomCode: string) => Promise<void>;
  onCreatePhone1Offer: () => Promise<string>;
  onAcceptOfferAndCreatePhone2Answer: (offerToken: string) => Promise<string>;
  onCompletePhone1Handshake: (answerToken: string) => Promise<void>;
  onDisconnect: () => void;
}

export const PairingPanel: React.FC<PairingPanelProps> = ({
  role,
  connectionState,
  stats,
  errorMessage,
  onSelectRole,
  onConnectLoopback,
  onStartBroadcastAutoPair,
  onCreatePhone1Offer,
  onAcceptOfferAndCreatePhone2Answer,
  onCompletePhone1Handshake,
  onDisconnect,
}) => {
  const [offerToken, setOfferToken] = useState('');
  const [remoteOfferInput, setRemoteOfferInput] = useState(() => {
    if (typeof window !== 'undefined' && window.location.hash.startsWith('#offer=')) {
      return decodeURIComponent(window.location.hash.slice(7));
    }
    return '';
  });
  const [answerToken, setAnswerToken] = useState('');
  const [remoteAnswerInput, setRemoteAnswerInput] = useState('');
  const [roomCode, setRoomCode] = useState('4242');
  const [localError, setLocalError] = useState<string | null>(null);
  const [copiedLabel, setCopiedLabel] = useState<string | null>(null);

  const handleCopy = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedLabel(label);
      setTimeout(() => setCopiedLabel(null), 1800);
    } catch {
      setLocalError('Clipboard write failed; please select and copy the token text manually.');
    }
  };

  const handleGenerateOffer = async () => {
    setLocalError(null);
    try {
      const token = await onCreatePhone1Offer();
      setOfferToken(token);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Failed to create SDP Offer.');
    }
  };

  const handleAcceptOffer = async () => {
    setLocalError(null);
    if (!remoteOfferInput.trim()) {
      setLocalError('Paste Phone 1 SDP Offer token before generating an Answer.');
      return;
    }
    try {
      const ans = await onAcceptOfferAndCreatePhone2Answer(remoteOfferInput.trim());
      setAnswerToken(ans);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Invalid SDP Offer token.');
    }
  };

  const handleFinalizeHandshake = async () => {
    setLocalError(null);
    if (!remoteAnswerInput.trim()) {
      setLocalError('Paste Phone 2 SDP Answer token to complete the P2P connection.');
      return;
    }
    try {
      await onCompletePhone1Handshake(remoteAnswerInput.trim());
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Failed to apply SDP Answer token.');
    }
  };

  const activeQrPayload =
    role === 'node-a'
      ? offerToken || 'SPLITBYTE-NODE-A-READY'
      : role === 'node-b'
      ? answerToken || 'SPLITBYTE-NODE-B-READY'
      : 'SPLITBYTE-DUAL-WEBRTC-LOOPBACK';
  const qrMatrix = generatePairingMatrix(activeQrPayload, 25);

  return (
    <div className="space-y-6">
      {/* Topology & Device Role Selector */}
      <div className="bg-[#111827] border border-slate-800 rounded-lg p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-slate-800">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              01. Device Role & P2P WebRTC DataChannel Topology
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Assign this browser as Phone 1 (Layers 0–2), Phone 2 (Layers 3–5), or Dual-Node P2P WebRTC Loopback
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono tabular-nums">
            <span className="text-slate-400">
              State: <strong className="text-slate-100 uppercase">{connectionState}</strong>
            </span>
            {connectionState !== 'disconnected' && (
              <button
                type="button"
                onClick={onDisconnect}
                className="px-3 py-1.5 rounded border border-slate-700 bg-slate-900 hover:bg-slate-800 text-slate-200 transition-colors whitespace-nowrap shrink-0"
              >
                Reset Channel
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Role 1: Dual-Node WebRTC Loopback */}
          <button
            type="button"
            onClick={() => onSelectRole('dual-loopback')}
            className={`text-left p-4 rounded-lg border transition-colors ${
              role === 'dual-loopback'
                ? 'border-cyan-400 bg-slate-900/90'
                : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between text-xs font-mono text-cyan-400 mb-1">
              <span>Dual-Node Loopback</span>
              <span>pcA ⇄ pcB</span>
            </div>
            <div className="text-sm font-semibold text-slate-100">
              Both Shards over Local WebRTC
            </div>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Spawns two isolated RTCPeerConnection peers (Node A Layers 0–2 and Node B Layers 3–5) connected via a real binary RTCDataChannel.
            </p>
          </button>

          {/* Role 2: Phone 1 (Node A) */}
          <button
            type="button"
            onClick={() => onSelectRole('node-a')}
            className={`text-left p-4 rounded-lg border transition-colors ${
              role === 'node-a'
                ? 'border-cyan-400 bg-slate-900/90'
                : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between text-xs font-mono text-cyan-400 mb-1">
              <span>Phone 1 · Caller</span>
              <span>22.47 MB FP32</span>
            </div>
            <div className="text-sm font-semibold text-slate-100">
              Node A (Embeddings + Layers 0–2)
            </div>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Holds only Shard A (5,618,304 params). Computes Layer 2 activations [1×384], streams Float32Array to Phone 2, and awaits 1-byte token replies.
            </p>
          </button>

          {/* Role 3: Phone 2 (Node B) */}
          <button
            type="button"
            onClick={() => onSelectRole('node-b')}
            className={`text-left p-4 rounded-lg border transition-colors ${
              role === 'node-b'
                ? 'border-amber-400 bg-slate-900/90'
                : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between text-xs font-mono text-amber-400 mb-1">
              <span>Phone 2 · Callee</span>
              <span>21.69 MB FP32</span>
            </div>
            <div className="text-sm font-semibold text-slate-100">
              Node B (Layers 3–5 + LM Head)
            </div>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Holds only Shard B (5,422,464 params). Receives Layer 2 activations from Phone 1, executes Layers 3–5 + LM Head, and returns next byte.
            </p>
          </button>
        </div>

        {(localError || errorMessage) && (
          <div
            role="alert"
            className="mt-4 p-3 rounded border border-rose-500/40 bg-rose-950/30 text-xs font-mono text-rose-200"
          >
            ✖ {localError || errorMessage}
          </div>
        )}
      </div>

      {/* Physical Two-Phone SDP Exchange & Multi-Tab Pairing Controls */}
      {role === 'dual-loopback' ? (
        <div className="bg-[#111827] border border-slate-800 rounded-lg p-5 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Local Dual-Peer WebRTC DataChannel Loopback Active
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Node A (pcA) and Node B (pcB) communicate strictly via binary ArrayBuffers over RTCDataChannel (&quot;tensor-mesh&quot;).
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onConnectLoopback}
              className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-colors whitespace-nowrap shrink-0"
            >
              {connectionState === 'connected' ? 'Reconnect WebRTC Loopback' : 'Connect WebRTC Loopback'}
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Left: Manual / QR SDP Handshake (Works 100% Offline Across Two Physical Phones) */}
          <div className="lg:col-span-8 bg-[#111827] border border-slate-800 rounded-lg p-5 space-y-4">
            <div className="border-b border-slate-800 pb-3">
              <h3 className="text-sm font-semibold text-slate-100">
                02. Direct Two-Phone WebRTC Handshake (Zero-Server SDP Exchange)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Establishes a direct UDP/SCTP RTCDataChannel between Phone 1 and Phone 2 with zero external relay servers
              </p>
            </div>

            {role === 'node-a' ? (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs font-medium text-slate-300">
                      Step 1 (Phone 1): Generate WebRTC SDP Offer Token
                    </label>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleGenerateOffer}
                        className="px-3 py-1.5 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-colors whitespace-nowrap shrink-0"
                      >
                        Generate SDP Offer
                      </button>
                      {offerToken && (
                        <>
                          <button
                            type="button"
                            onClick={() => handleCopy(offerToken, 'offer')}
                            className="px-3 py-1.5 rounded border border-slate-700 bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-mono transition-colors whitespace-nowrap shrink-0"
                          >
                            {copiedLabel === 'offer' ? 'Copied Offer' : 'Copy Offer Token'}
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              handleCopy(
                                `${window.location.origin}${window.location.pathname}#offer=${encodeURIComponent(offerToken)}`,
                                'join-link'
                              )
                            }
                            className="px-3 py-1.5 rounded border border-cyan-500/50 bg-cyan-950/30 hover:bg-cyan-900/40 text-cyan-200 text-xs font-mono transition-colors whitespace-nowrap shrink-0"
                          >
                            {copiedLabel === 'join-link' ? 'Copied Link' : 'Copy Phone 2 Link'}
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  <textarea
                    readOnly
                    aria-label="Phone 1 Generated SDP Offer Token"
                    value={offerToken}
                    placeholder="Click 'Generate SDP Offer' to gather local ICE candidates and create a compact pairing token for Phone 2..."
                    className="w-full h-20 rounded bg-slate-950 border border-slate-800 p-2.5 text-xs font-mono text-cyan-300 focus:outline-none"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label htmlFor="phone2-answer-input" className="text-xs font-medium text-slate-300">
                      Step 2 (Phone 1): Paste SDP Answer Token from Phone 2
                    </label>
                    <button
                      type="button"
                      onClick={handleFinalizeHandshake}
                      className="px-3 py-1.5 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs transition-colors whitespace-nowrap shrink-0"
                    >
                      Complete P2P Handshake
                    </button>
                  </div>
                  <textarea
                    id="phone2-answer-input"
                    value={remoteAnswerInput}
                    onChange={(e) => setRemoteAnswerInput(e.target.value)}
                    placeholder="Paste the base64 SDP Answer token generated on Phone 2..."
                    className="w-full h-20 rounded bg-slate-950 border border-slate-800 p-2.5 text-xs font-mono text-slate-200 focus:border-cyan-500 focus:outline-none"
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label htmlFor="phone1-offer-input" className="text-xs font-medium text-slate-300">
                      Step 1 (Phone 2): Paste SDP Offer Token from Phone 1
                    </label>
                    <button
                      type="button"
                      onClick={handleAcceptOffer}
                      className="px-3 py-1.5 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-xs transition-colors whitespace-nowrap shrink-0"
                    >
                      Accept Offer &amp; Create Answer
                    </button>
                  </div>
                  <textarea
                    id="phone1-offer-input"
                    value={remoteOfferInput}
                    onChange={(e) => setRemoteOfferInput(e.target.value)}
                    placeholder="Paste the SDP Offer token generated by Phone 1..."
                    className="w-full h-20 rounded bg-slate-950 border border-slate-800 p-2.5 text-xs font-mono text-slate-200 focus:border-amber-500 focus:outline-none"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs font-medium text-slate-300">
                      Step 2 (Phone 2): Copy SDP Answer Token Back to Phone 1
                    </label>
                    {answerToken && (
                      <button
                        type="button"
                        onClick={() => handleCopy(answerToken, 'answer')}
                        className="px-3 py-1.5 rounded border border-slate-700 bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-mono transition-colors whitespace-nowrap shrink-0"
                      >
                        {copiedLabel === 'answer' ? 'Copied Answer' : 'Copy Answer Token'}
                      </button>
                    )}
                  </div>
                  <textarea
                    readOnly
                    aria-label="Phone 2 Generated SDP Answer Token"
                    value={answerToken}
                    placeholder="Generated SDP Answer token will appear here..."
                    className="w-full h-20 rounded bg-slate-950 border border-slate-800 p-2.5 text-xs font-mono text-amber-300 focus:outline-none"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Right: Optical Matrix + Same-Device Multi-Tab Auto-Signaling */}
          <div className="lg:col-span-4 bg-[#111827] border border-slate-800 rounded-lg p-5 flex flex-col justify-between space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                1-Tap Two-Phone Room Pairing &amp; Optical Matrix
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Enter the same 4-digit Room Code on both phones for automatic WebRTC SDP exchange
              </p>

              <div className="mt-4 flex flex-col items-center justify-center p-4 bg-slate-950 rounded border border-slate-800">
                <svg
                  viewBox="0 0 25 25"
                  className="w-36 h-36 bg-white p-2 rounded"
                  role="img"
                  aria-label="Offline SDP Pairing Matrix"
                >
                  {qrMatrix.map((row, rIdx) =>
                    row.map((cell, cIdx) =>
                      cell ? (
                        <rect
                          key={`${rIdx}-${cIdx}`}
                          x={cIdx}
                          y={rIdx}
                          width={1}
                          height={1}
                          fill="#0B0E17"
                        />
                      ) : null
                    )
                  )}
                </svg>
                <span className="mt-2 text-[11px] font-mono text-slate-400 tabular-nums">
                  {role === 'node-a'
                    ? offerToken
                      ? `SDP Offer (${offerToken.length} chars)`
                      : 'Room #' + roomCode + ' Ready'
                    : answerToken
                    ? `SDP Answer (${answerToken.length} chars)`
                    : 'Room #' + roomCode + ' Ready'}
                </span>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-800 space-y-2.5">
              <label htmlFor="pairing-room-code" className="block text-xs font-medium text-slate-300">
                Two-Phone Auto-Handshake Room Code
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="pairing-room-code"
                  type="text"
                  value={roomCode}
                  onChange={(e) => setRoomCode(e.target.value)}
                  className="w-24 px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-100 tabular-nums"
                />
                <button
                  type="button"
                  onClick={() => onStartBroadcastAutoPair(role, roomCode)}
                  className="flex-1 px-3 py-1.5 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-semibold transition-colors whitespace-nowrap shrink-0"
                >
                  Auto-Pair Room #{roomCode}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
