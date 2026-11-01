import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface RoomSignalState {
  offerSdp: string | null;
  answerSdp: string | null;
  updatedAt: number;
}

// Ephemeral in-memory WebRTC SDP handshake rooms (strictly for initial P2P Offer/Answer exchange;
// zero tensor activations or tokens ever touch this server).
const signalRooms = new Map<string, RoomSignalState>();

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: '256kb' }));

  // Ephemeral WebRTC SDP Offer/Answer Rendezvous Endpoint for 1-Tap Two-Phone Pairing
  app.get('/api/signal/:room', (req, res) => {
    const room = String(req.params.room || '4242').trim();
    const entry = signalRooms.get(room) || { offerSdp: null, answerSdp: null, updatedAt: Date.now() };
    res.json(entry);
  });

  app.post('/api/signal/:room', (req, res) => {
    const room = String(req.params.room || '4242').trim();
    const { type, sdp } = req.body || {};
    const existing = signalRooms.get(room) || {
      offerSdp: null,
      answerSdp: null,
      updatedAt: Date.now(),
    };

    if (type === 'reset') {
      signalRooms.delete(room);
      res.json({ ok: true });
      return;
    }

    if (type === 'offer' && typeof sdp === 'string') {
      signalRooms.set(room, {
        offerSdp: sdp,
        answerSdp: null,
        updatedAt: Date.now(),
      });
      res.json({ ok: true });
      return;
    }

    if (type === 'answer' && typeof sdp === 'string') {
      signalRooms.set(room, {
        offerSdp: existing.offerSdp,
        answerSdp: sdp,
        updatedAt: Date.now(),
      });
      res.json({ ok: true });
      return;
    }

    res.status(400).json({ error: 'Invalid signal payload' });
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
