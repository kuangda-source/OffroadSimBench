# Dashboard

Local control panel for the first-stage benchmark loop.

- `backend/`: FastAPI service for catalogs, backend status, episode runs, and saved episode metrics.
- `frontend/`: React + Vite + TypeScript app that calls the local API.

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
