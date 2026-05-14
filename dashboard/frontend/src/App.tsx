import { Activity, Database, Play, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type CatalogItem = {
  id?: string;
  name?: string;
  path?: string;
  available?: boolean;
  message?: string;
  description?: string;
};

type RunResponse = {
  episode_id: string;
  scenario_id: string;
  backend: string;
  agent: string;
  steps: number;
  done: boolean;
  episode_path: string | null;
  metrics: Record<string, unknown>;
};

type EpisodeSummary = {
  episode_id: string;
  path: string;
  metadata: Record<string, unknown>;
  metrics: Record<string, unknown>;
};

type EpisodeDetail = EpisodeSummary & {
  steps_preview: Array<{
    observation?: {
      vehicle_state?: {
        x: number;
        y: number;
        speed?: number;
      };
    };
  }>;
};

type CatalogState = {
  health: string;
  scenarios: CatalogItem[];
  vehicles: CatalogItem[];
  agents: CatalogItem[];
  backends: CatalogItem[];
  episodes: EpisodeSummary[];
};

const emptyCatalog: CatalogState = {
  health: "offline",
  scenarios: [],
  vehicles: [],
  agents: [],
  backends: [],
  episodes: []
};

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || response.statusText);
  }
  return response.json() as Promise<T>;
}

function App() {
  const [catalog, setCatalog] = useState<CatalogState>(emptyCatalog);
  const [selected, setSelected] = useState({
    backend: "gym_heightmap",
    scenario: "forest_trail_001",
    agent: "rule_based",
    seed: 7,
    maxSteps: 250,
    record: true
  });
  const [runResult, setRunResult] = useState<RunResponse | null>(null);
  const [episodeDetail, setEpisodeDetail] = useState<EpisodeDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setError(null);
    try {
      const [health, scenarios, vehicles, agents, backends, episodes] = await Promise.all([
        fetchJson<{ status: string }>("/health"),
        fetchJson<CatalogItem[]>("/scenarios"),
        fetchJson<CatalogItem[]>("/vehicles"),
        fetchJson<CatalogItem[]>("/agents"),
        fetchJson<CatalogItem[]>("/backends"),
        fetchJson<EpisodeSummary[]>("/episodes")
      ]);
      setCatalog({
        health: health.status,
        scenarios,
        vehicles,
        agents,
        backends,
        episodes
      });
    } catch (err) {
      setCatalog((current) => ({ ...current, health: "offline" }));
      setError(err instanceof Error ? err.message : "Failed to reach dashboard API.");
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const runEpisode = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<RunResponse>("/run_episode", {
        method: "POST",
        body: JSON.stringify({
          backend: selected.backend,
          scenario: selected.scenario,
          agent: selected.agent,
          seed: selected.seed,
          max_steps: selected.maxSteps,
          record: selected.record
        })
      });
      setRunResult(result);
      if (result.episode_path) {
        const detail = await fetchJson<EpisodeDetail>(`/episodes/${result.episode_id}`);
        setEpisodeDetail(detail);
      } else {
        setEpisodeDetail(null);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Episode run failed.");
    } finally {
      setBusy(false);
    }
  };

  const metrics = runResult?.metrics ?? episodeDetail?.metrics ?? {};
  const selectedBackend = catalog.backends.find((item) => item.name === selected.backend);

  return (
    <main className="app-shell">
      <aside className="control-rail">
        <div className="brand-row">
          <Activity size={22} aria-hidden="true" />
          <div>
            <h1>OffroadSimBench</h1>
            <span className={catalog.health === "ok" ? "status-ok" : "status-bad"}>{catalog.health}</span>
          </div>
        </div>

        <label>
          Backend
          <select
            value={selected.backend}
            onChange={(event) => setSelected({ ...selected, backend: event.target.value })}
          >
            {catalog.backends.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          Scenario
          <select
            value={selected.scenario}
            onChange={(event) => setSelected({ ...selected, scenario: event.target.value })}
          >
            {catalog.scenarios.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id}
              </option>
            ))}
          </select>
        </label>

        <label>
          Agent
          <select
            value={selected.agent}
            onChange={(event) => setSelected({ ...selected, agent: event.target.value })}
          >
            {catalog.agents.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
        </label>

        <div className="number-grid">
          <label>
            Seed
            <input
              type="number"
              value={selected.seed}
              onChange={(event) => setSelected({ ...selected, seed: Number(event.target.value) })}
            />
          </label>
          <label>
            Steps
            <input
              type="number"
              min={1}
              max={100000}
              value={selected.maxSteps}
              onChange={(event) => setSelected({ ...selected, maxSteps: Number(event.target.value) })}
            />
          </label>
        </div>

        <label className="toggle-row">
          <input
            type="checkbox"
            checked={selected.record}
            onChange={(event) => setSelected({ ...selected, record: event.target.checked })}
          />
          Record episode
        </label>

        <div className="button-row">
          <button type="button" className="primary-button" onClick={runEpisode} disabled={busy} title="Run episode">
            <Play size={18} aria-hidden="true" />
            <span>{busy ? "Running" : "Run"}</span>
          </button>
          <button type="button" className="icon-button" onClick={refresh} title="Refresh catalogs">
            <RefreshCw size={18} aria-hidden="true" />
          </button>
        </div>

        {error && <p className="error-text">{error}</p>}
      </aside>

      <section className="workspace">
        <header className="top-strip">
          <div>
            <span className="eyebrow">Active Backend</span>
            <h2>{selected.backend}</h2>
            <p>{selectedBackend?.description ?? "No backend selected"}</p>
          </div>
          <div className="top-metrics">
            <MetricBadge label="Episodes" value={catalog.episodes.length} />
            <MetricBadge label="Steps" value={String(runResult?.steps ?? "-")} />
            <MetricBadge label="Done" value={runResult ? String(runResult.done) : "-"} />
          </div>
        </header>

        <div className="main-grid">
          <section className="panel trajectory-panel">
            <div className="panel-heading">
              <h3>Trajectory</h3>
              <span>{episodeDetail?.episode_id ?? runResult?.episode_id ?? "pending"}</span>
            </div>
            <TrajectoryPlot detail={episodeDetail} />
          </section>

          <section className="panel metrics-panel">
            <div className="panel-heading">
              <h3>Metrics</h3>
              <Database size={18} aria-hidden="true" />
            </div>
            <MetricTable metrics={metrics} />
          </section>
        </div>

        <section className="panel history-panel">
          <div className="panel-heading">
            <h3>Recent Episodes</h3>
            <span>{catalog.episodes.length}</span>
          </div>
          <EpisodeList
            episodes={catalog.episodes}
            onSelect={async (episodeId) => {
              setError(null);
              try {
                setEpisodeDetail(await fetchJson<EpisodeDetail>(`/episodes/${episodeId}`));
              } catch (err) {
                setError(err instanceof Error ? err.message : "Could not load episode.");
              }
            }}
          />
        </section>
      </section>
    </main>
  );
}

function MetricBadge({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric-badge">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricTable({ metrics }: { metrics: Record<string, unknown> }) {
  const rows = Object.entries(metrics).slice(0, 14);
  if (!rows.length) {
    return <div className="empty-state">No metrics yet</div>;
  }
  return (
    <table>
      <tbody>
        {rows.map(([key, value]) => (
          <tr key={key}>
            <th>{key}</th>
            <td>{formatValue(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EpisodeList({
  episodes,
  onSelect
}: {
  episodes: EpisodeSummary[];
  onSelect: (episodeId: string) => void;
}) {
  if (!episodes.length) {
    return <div className="empty-state">No recorded episodes</div>;
  }
  return (
    <div className="episode-list">
      {episodes.slice(0, 8).map((episode) => (
        <button key={episode.episode_id} type="button" onClick={() => onSelect(episode.episode_id)}>
          <span>{episode.episode_id}</span>
          <strong>{formatValue(episode.metrics.success ?? "-")}</strong>
        </button>
      ))}
    </div>
  );
}

function TrajectoryPlot({ detail }: { detail: EpisodeDetail | null }) {
  const points = useMemo(() => {
    return (
      detail?.steps_preview
        .map((step) => step.observation?.vehicle_state)
        .filter((state): state is { x: number; y: number; speed?: number } => Boolean(state)) ?? []
    );
  }, [detail]);

  if (!points.length) {
    return <div className="plot-empty">Run or select a recorded episode</div>;
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = 720;
  const height = 360;
  const pad = 32;
  const scaleX = (x: number) => pad + ((x - minX) / Math.max(maxX - minX, 1)) * (width - pad * 2);
  const scaleY = (y: number) => height - pad - ((y - minY) / Math.max(maxY - minY, 1)) * (height - pad * 2);
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Episode trajectory">
      <rect x="0" y="0" width={width} height={height} rx="0" />
      <path d={path} />
      <circle cx={scaleX(points[0].x)} cy={scaleY(points[0].y)} r="6" className="start-point" />
      <circle cx={scaleX(points[points.length - 1].x)} cy={scaleY(points[points.length - 1].y)} r="6" className="end-point" />
    </svg>
  );
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export default App;
