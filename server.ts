import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface QueuedBinaryFrame {
  seq: number;
  frameId?: string;
  to: 'node-a' | 'node-b';
  b64: string;
  ts: number;
}

interface RoomSignalState {
  offerSdp: string | null;
  answerSdp: string | null;
  nodeAPingAt: number;
  nodeBPingAt: number;
  nextSeq: number;
  frames: QueuedBinaryFrame[];
  updatedAt: number;
}

const signalRooms = new Map<string, RoomSignalState>();

function getOrCreateRoom(room: string): RoomSignalState {
  let entry = signalRooms.get(room);
  if (!entry) {
    entry = {
      offerSdp: null,
      answerSdp: null,
      nodeAPingAt: 0,
      nodeBPingAt: 0,
      nextSeq: 1,
      frames: [],
      updatedAt: Date.now(),
    };
    signalRooms.set(room, entry);
  }
  return entry;
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: '4mb' }));

  app.get('/api/signal/:room', (req, res) => {
    const room = String(req.params.room || '4242').trim();
    const role = String(req.query.role || '');
    const sinceSeq = Number(req.query.since || 0);
    const entry = getOrCreateRoom(room);
    const now = Date.now();

    if (role === 'node-a') entry.nodeAPingAt = now;
    if (role === 'node-b') entry.nodeBPingAt = now;

    // Prune frames older than 30 seconds
    entry.frames = entry.frames.filter((f) => now - f.ts < 30000);

    const incomingFrames =
      role === 'node-a' || role === 'node-b'
        ? entry.frames.filter((f) => f.to === role && f.seq > sinceSeq)
        : [];

    res.json({
      offerSdp: entry.offerSdp,
      answerSdp: entry.answerSdp,
      nodeAOnline: now - entry.nodeAPingAt < 6000,
      nodeBOnline: now - entry.nodeBPingAt < 6000,
      frames: incomingFrames,
      latestSeq: entry.nextSeq - 1,
    });
  });

  app.post('/api/signal/:room', (req, res) => {
    const room = String(req.params.room || '4242').trim();
    const { type, sdp, to, b64, role, frameId } = req.body || {};
    const entry = getOrCreateRoom(room);
    const now = Date.now();

    if (role === 'node-a') entry.nodeAPingAt = now;
    if (role === 'node-b') entry.nodeBPingAt = now;

    if (type === 'reset') {
      signalRooms.delete(room);
      res.json({ ok: true });
      return;
    }

    if (type === 'offer' && typeof sdp === 'string') {
      entry.offerSdp = sdp;
      entry.answerSdp = null;
      entry.updatedAt = now;
      res.json({ ok: true });
      return;
    }

    if (type === 'answer' && typeof sdp === 'string') {
      entry.answerSdp = sdp;
      entry.updatedAt = now;
      res.json({ ok: true });
      return;
    }

    if (type === 'frame' && (to === 'node-a' || to === 'node-b') && typeof b64 === 'string') {
      const seq = entry.nextSeq++;
      entry.frames.push({ seq, frameId: typeof frameId === 'string' ? frameId : undefined, to, b64, ts: now });
      if (entry.frames.length > 250) {
        entry.frames.splice(0, entry.frames.length - 250);
      }
      res.json({ ok: true, seq });
      return;
    }

    res.json({ ok: true });
  });

  const distPath = path.join(__dirname, 'dist');
  const hasDist = fs.existsSync(path.join(distPath, 'index.html'));
  const isDev = process.env.NODE_ENV === 'development' || !hasDist;

  if (isDev) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`SplitByte P2P server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
