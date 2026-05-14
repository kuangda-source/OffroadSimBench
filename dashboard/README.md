# Dashboard

Local control panel for the benchmark loop and second-stage visual demo.

- `backend/`: FastAPI service for catalogs, backend status, episode runs, SSE streaming, saved steps, and saved episode metrics.
- `frontend/`: React + Vite + TypeScript app for live terrain/BEV visualization and replay.

Start the API from the repo root:

```powershell
uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
cd dashboard/frontend
npm install
npm run dev
```

Build the frontend:

```powershell
npm run build
```

Useful API endpoints:

- `GET /stream_episode`
- `GET /episodes/{episode_id}/steps`
- `GET /beamng/status`
