import {
  Activity,
  Database,
  Gauge,
  Map,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  SkipBack,
  Square,
  StepForward,
  Waypoints
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type CatalogItem = {
  id?: string;
  name?: string;
  path?: string;
  available?: boolean;
  message?: string;
  description?: string;
  details?: Record<string, unknown>;
};

type ArrayLayers = {
  shape: number[];
  layers: Record<string, number[][]>;
};

type VehicleState = {
  x: number;
  y: number;
  z?: number;
  yaw?: number;
  pitch?: number;
  roll?: number;
  speed?: number;
};

type ObservationPayload = {
  timestamp: number;
  vehicle_state: VehicleState;
  goal: number[];
  info: Record<string, unknown>;
  local_bev?: ArrayLayers | null;
  terrain_map?: ArrayLayers | null;
};

type ActionPayload = {
  steer: number;
  throttle: number;
  brake: number;
} | null;

type FramePayload = {
  step_index: number;
  observation: ObservationPayload;
  action: ActionPayload;
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
};

type StreamPayload = {
  event: "start" | "step" | "end" | "error";
  episode_id?: string;
  scenario_id?: string;
  backend?: string;
  agent?: string;
  frame?: FramePayload;
  metrics?: Record<string, unknown>;
  episode_path?: string | null;
  detail?: string;
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
      vehicle_state?: VehicleState;
    };
  }>;
};

type EpisodeStep = {
  step_index: number;
  observation: ObservationPayload;
  action: ActionPayload;
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
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
  const eventSourceRef = useRef<EventSource | null>(null);
  const [catalog, setCatalog] = useState<CatalogState>(emptyCatalog);
  const [selected, setSelected] = useState({
    backend: "gym_heightmap",
    scenario: "forest_trail_001",
    agent: "rule_based",
    seed: 7,
    maxSteps: 240,
    record: true
  });
  const [frames, setFrames] = useState<FramePayload[]>([]);
  const [replayFrames, setReplayFrames] = useState<FramePayload[]>([]);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [streamMetrics, setStreamMetrics] = useState<Record<string, unknown>>({});
  const [episodeDetail, setEpisodeDetail] = useState<EpisodeDetail | null>(null);
  const [liveEpisodeId, setLiveEpisodeId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
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
    return () => eventSourceRef.current?.close();
  }, []);

  const activeFrames = replayFrames.length ? replayFrames : frames;
  const currentFrame = activeFrames[Math.min(selectedFrameIndex, Math.max(activeFrames.length - 1, 0))] ?? null;
  const activeMetrics = Object.keys(streamMetrics).length ? streamMetrics : (episodeDetail?.metrics ?? {});
  const selectedBackend = catalog.backends.find((item) => item.name === selected.backend);
  const beamngBackend = catalog.backends.find((item) => item.name === "beamng");
  const canStream = selected.backend === "gym_heightmap";

  useEffect(() => {
    if (isStreaming && frames.length) {
      setSelectedFrameIndex(frames.length - 1);
    }
  }, [frames.length, isStreaming]);

  useEffect(() => {
    if (!isPlaying || activeFrames.length < 2) {
      return undefined;
    }
    const handle = window.setInterval(() => {
      setSelectedFrameIndex((current) => {
        if (current >= activeFrames.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 120);
    return () => window.clearInterval(handle);
  }, [activeFrames.length, isPlaying]);

  const startStream = () => {
    if (!canStream) {
      setError("Streaming run currently targets the local gym_heightmap backend.");
      return;
    }
    eventSourceRef.current?.close();
    setError(null);
    setFrames([]);
    setReplayFrames([]);
    setEpisodeDetail(null);
    setStreamMetrics({});
    setLiveEpisodeId(null);
    setSelectedFrameIndex(0);
    setIsPlaying(false);
    setIsStreaming(true);

    const params = new URLSearchParams({
      backend: selected.backend,
      scenario: selected.scenario,
      agent: selected.agent,
      seed: String(selected.seed),
      max_steps: String(selected.maxSteps),
      record: String(selected.record),
      record_arrays: "true",
      delay_ms: "35"
    });
    const source = new EventSource(`${API_BASE}/stream_episode?${params.toString()}`);
    eventSourceRef.current = source;

    const parseEvent = (event: Event) => JSON.parse((event as MessageEvent).data) as StreamPayload;
    source.addEventListener("start", (event) => {
      const payload = parseEvent(event);
      setLiveEpisodeId(payload.episode_id ?? null);
      setFrames(payload.frame ? [payload.frame] : []);
    });
    source.addEventListener("step", (event) => {
      const payload = parseEvent(event);
      if (payload.frame) {
        setFrames((current) => [...current, payload.frame as FramePayload]);
      }
      if (payload.metrics) {
        setStreamMetrics(payload.metrics);
      }
    });
    source.addEventListener("end", (event) => {
      const payload = parseEvent(event);
      if (payload.metrics) {
        setStreamMetrics(payload.metrics);
      }
      setIsStreaming(false);
      source.close();
      void refresh();
    });
    source.addEventListener("error", (event) => {
      const payload = "data" in event ? parseEvent(event) : null;
      setError(payload?.detail ?? "Episode stream closed unexpectedly.");
      setIsStreaming(false);
      source.close();
    });
  };

  const stopStream = () => {
    eventSourceRef.current?.close();
    setIsStreaming(false);
  };

  const loadEpisode = async (episodeId: string) => {
    setError(null);
    setIsPlaying(false);
    try {
      const [detail, steps] = await Promise.all([
        fetchJson<EpisodeDetail>(`/episodes/${episodeId}`),
        fetchJson<EpisodeStep[]>(`/episodes/${episodeId}/steps?limit=5000&include_arrays=true`)
      ]);
      setEpisodeDetail(detail);
      setLiveEpisodeId(episodeId);
      setReplayFrames(steps.map(stepToFrame));
      setStreamMetrics(detail.metrics ?? {});
      setSelectedFrameIndex(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load episode.");
    }
  };

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

        <label className="range-control">
          Step budget
          <input
            type="range"
            min={20}
            max={1200}
            step={20}
            value={Math.min(selected.maxSteps, 1200)}
            onChange={(event) => setSelected({ ...selected, maxSteps: Number(event.target.value) })}
          />
        </label>

        <label className="toggle-row">
          <input
            type="checkbox"
            checked={selected.record}
            onChange={(event) => setSelected({ ...selected, record: event.target.checked })}
          />
          Record episode
        </label>

        <div className="button-row">
          <button
            type="button"
            className="primary-button"
            onClick={startStream}
            disabled={isStreaming || !canStream}
            title="Start streaming run"
          >
            <Play size={18} aria-hidden="true" />
            <span>Stream</span>
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={stopStream}
            disabled={!isStreaming}
            title="Stop stream"
          >
            <Square size={18} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button" onClick={refresh} title="Refresh catalogs">
            <RefreshCw size={18} aria-hidden="true" />
          </button>
        </div>

        {error && <p className="error-text">{error}</p>}

        <BackendStatusPanel backend={selectedBackend} beamng={beamngBackend} />
      </aside>

      <section className="workspace">
        <header className="top-strip">
          <div>
            <span className="eyebrow">Active Episode</span>
            <h2>{liveEpisodeId ?? "standby"}</h2>
            <p>{selectedBackend?.description ?? "No backend selected"}</p>
          </div>
          <div className="top-metrics">
            <MetricBadge label="Frames" value={activeFrames.length} />
            <MetricBadge label="Step" value={currentFrame?.step_index ?? "-"} />
            <MetricBadge label="Speed" value={formatNumber(currentFrame?.observation.vehicle_state.speed)} />
            <MetricBadge label="Risk" value={formatNumber(currentFrame?.observation.info.terrain_risk as number)} />
          </div>
        </header>

        <div className="visual-grid">
          <section className="panel map-panel">
            <div className="panel-heading">
              <div>
                <h3>Terrain Risk</h3>
                <span>{selected.scenario}</span>
              </div>
              <Map size={18} aria-hidden="true" />
            </div>
            <TerrainPanel frames={activeFrames} frameIndex={selectedFrameIndex} />
          </section>

          <section className="panel side-panel">
            <div className="panel-heading">
              <div>
                <h3>Local BEV</h3>
                <span>step {currentFrame?.step_index ?? 0}</span>
              </div>
              <Gauge size={18} aria-hidden="true" />
            </div>
            <Heatmap payload={currentFrame?.observation.local_bev ?? null} layer="risk" compact />
            <MetricTable metrics={activeMetrics} />
          </section>
        </div>

        <section className="panel replay-panel">
          <div className="panel-heading">
            <div>
              <h3>Replay</h3>
              <span>{activeFrames.length ? `${selectedFrameIndex + 1}/${activeFrames.length}` : "0/0"}</span>
            </div>
            <Waypoints size={18} aria-hidden="true" />
          </div>
          <ReplayControls
            frameCount={activeFrames.length}
            frameIndex={selectedFrameIndex}
            isPlaying={isPlaying}
            onChange={setSelectedFrameIndex}
            onReset={() => setSelectedFrameIndex(0)}
            onTogglePlay={() => setIsPlaying((current) => !current)}
            onStepForward={() => setSelectedFrameIndex((current) => Math.min(current + 1, activeFrames.length - 1))}
            onStepBack={() => setSelectedFrameIndex((current) => Math.max(current - 1, 0))}
          />
        </section>

        <section className="panel history-panel">
          <div className="panel-heading">
            <div>
              <h3>Recent Episodes</h3>
              <span>{catalog.episodes.length}</span>
            </div>
            <Database size={18} aria-hidden="true" />
          </div>
          <EpisodeList episodes={catalog.episodes} onSelect={(episodeId) => void loadEpisode(episodeId)} />
        </section>
      </section>
    </main>
  );
}

function BackendStatusPanel({ backend, beamng }: { backend?: CatalogItem; beamng?: CatalogItem }) {
  return (
    <div className="runtime-panel">
      <div>
        <span>Selected</span>
        <strong className={backend?.available === false ? "status-bad" : "status-ok"}>
          {backend?.available === false ? "unavailable" : "available"}
        </strong>
      </div>
      <div>
        <span>BeamNG</span>
        <strong className={beamng?.available ? "status-ok" : "status-warn"}>
          {beamng?.available ? "ready" : "pending"}
        </strong>
      </div>
      <p>{beamng?.message ?? "Backend catalogs have not loaded yet."}</p>
    </div>
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
  const preferred = [
    "success",
    "done",
    "steps",
    "total_reward",
    "distance_to_goal",
    "average_speed",
    "max_speed",
    "collision_count",
    "path_length",
    "average_terrain_risk",
    "control_smoothness"
  ];
  const rows = preferred
    .filter((key) => key in metrics)
    .map((key) => [key, metrics[key]] as const);
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

function TerrainPanel({ frames, frameIndex }: { frames: FramePayload[]; frameIndex: number }) {
  const terrain = latestArray(frames, frameIndex, "terrain_map");
  const trajectory = frames.slice(0, frameIndex + 1).map((frame) => frame.observation.vehicle_state);
  const goal = frames[Math.min(frameIndex, Math.max(frames.length - 1, 0))]?.observation.goal ?? [0, 0];

  return (
    <div className="terrain-stage">
      <Heatmap payload={terrain} layer="risk" />
      <TrajectoryOverlay points={trajectory} goal={goal} />
    </div>
  );
}

function Heatmap({
  payload,
  layer,
  compact = false
}: {
  payload?: ArrayLayers | null;
  layer: string;
  compact?: boolean;
}) {
  const matrix = payload?.layers[layer] ?? firstLayer(payload);
  if (!matrix?.length || !matrix[0]?.length) {
    return <div className={compact ? "empty-state compact" : "empty-state"}>No map frame</div>;
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const values = matrix.flat();
  const min = Math.min(...values);
  const max = Math.max(...values);
  return (
    <div
      className={compact ? "heatmap compact-heatmap" : "heatmap"}
      style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, aspectRatio: `${cols} / ${rows}` }}
    >
      {values.map((value, index) => (
        <span key={`${index}-${value}`} style={{ backgroundColor: heatColor(value, min, max) }} />
      ))}
    </div>
  );
}

function TrajectoryOverlay({ points, goal }: { points: VehicleState[]; goal: number[] }) {
  if (!points.length) {
    return null;
  }
  const extent = Math.max(128, goal[0] ?? 0, goal[1] ?? 0, ...points.map((point) => point.x), ...points.map((point) => point.y));
  const toX = (x: number) => (x / extent) * 100;
  const toY = (y: number) => 100 - (y / extent) * 100;
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${toX(point.x)} ${toY(point.y)}`).join(" ");
  const current = points[points.length - 1];

  return (
    <svg className="trajectory-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <path d={path} />
      <circle cx={toX(points[0].x)} cy={toY(points[0].y)} r="1.6" className="start-point" />
      <circle cx={toX(goal[0] ?? 0)} cy={toY(goal[1] ?? 0)} r="1.8" className="goal-point" />
      <circle cx={toX(current.x)} cy={toY(current.y)} r="1.5" className="end-point" />
    </svg>
  );
}

function ReplayControls({
  frameCount,
  frameIndex,
  isPlaying,
  onChange,
  onReset,
  onTogglePlay,
  onStepBack,
  onStepForward
}: {
  frameCount: number;
  frameIndex: number;
  isPlaying: boolean;
  onChange: (index: number) => void;
  onReset: () => void;
  onTogglePlay: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
}) {
  const disabled = frameCount === 0;
  return (
    <div className="replay-controls">
      <button type="button" className="icon-button light" onClick={onReset} disabled={disabled} title="Reset replay">
        <RotateCcw size={17} aria-hidden="true" />
      </button>
      <button type="button" className="icon-button light" onClick={onStepBack} disabled={disabled} title="Previous frame">
        <SkipBack size={17} aria-hidden="true" />
      </button>
      <button type="button" className="icon-button light" onClick={onTogglePlay} disabled={disabled} title="Play or pause replay">
        {isPlaying ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
      </button>
      <button type="button" className="icon-button light" onClick={onStepForward} disabled={disabled} title="Next frame">
        <StepForward size={17} aria-hidden="true" />
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(frameCount - 1, 0)}
        value={Math.min(frameIndex, Math.max(frameCount - 1, 0))}
        onChange={(event) => onChange(Number(event.target.value))}
        disabled={disabled}
      />
    </div>
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
      {episodes.slice(0, 10).map((episode) => (
        <button key={episode.episode_id} type="button" onClick={() => onSelect(episode.episode_id)}>
          <span>{episode.episode_id}</span>
          <strong>{formatValue(episode.metrics.success ?? episode.metrics.done ?? "-")}</strong>
        </button>
      ))}
    </div>
  );
}

function stepToFrame(step: EpisodeStep): FramePayload {
  return {
    step_index: step.step_index,
    observation: step.observation,
    action: step.action,
    reward: step.reward,
    done: step.done,
    info: step.info ?? {}
  };
}

function latestArray(
  frames: FramePayload[],
  frameIndex: number,
  key: "terrain_map" | "local_bev"
): ArrayLayers | null {
  for (let index = Math.min(frameIndex, frames.length - 1); index >= 0; index -= 1) {
    const payload = frames[index]?.observation[key];
    if (payload) {
      return payload;
    }
  }
  return null;
}

function firstLayer(payload?: ArrayLayers | null): number[][] | null {
  const firstKey = Object.keys(payload?.layers ?? {})[0];
  return firstKey ? (payload?.layers[firstKey] ?? null) : null;
}

function heatColor(value: number, min: number, max: number): string {
  const normalized = max <= min ? 0 : (value - min) / (max - min);
  if (normalized < 0.5) {
    const t = normalized / 0.5;
    return mixColor([35, 118, 86], [219, 169, 69], t);
  }
  return mixColor([219, 169, 69], [185, 65, 58], (normalized - 0.5) / 0.5);
}

function mixColor(from: number[], to: number[], t: number): string {
  const channels = from.map((value, index) => Math.round(value + (to[index] - value) * t));
  return `rgb(${channels[0]}, ${channels[1]}, ${channels[2]})`;
}

function formatNumber(value: number | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(2);
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
