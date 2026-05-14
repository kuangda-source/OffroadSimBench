import {
  Activity,
  Database,
  Gauge,
  Languages,
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

type Language = "en" | "zh";

type Copy = {
  labels: {
    language: string;
    backend: string;
    scenario: string;
    agent: string;
    seed: string;
    steps: string;
    stepBudget: string;
    recordEpisode: string;
    stream: string;
    selected: string;
    beamng: string;
    activeEpisode: string;
    frames: string;
    step: string;
    speed: string;
    risk: string;
    terrainRisk: string;
    localBev: string;
    replay: string;
    recentEpisodes: string;
  };
  status: {
    ok: string;
    offline: string;
    available: string;
    unavailable: string;
    ready: string;
    pending: string;
    standby: string;
  };
  messages: {
    noBackend: string;
    noCatalog: string;
    noMetrics: string;
    noMapFrame: string;
    noEpisodes: string;
    apiError: string;
    streamBackendOnly: string;
    streamClosed: string;
    loadEpisodeFailed: string;
  };
  titles: {
    startStream: string;
    stopStream: string;
    refresh: string;
    resetReplay: string;
    previousFrame: string;
    playPause: string;
    nextFrame: string;
  };
  metrics: Record<string, string>;
};

const copyByLanguage: Record<Language, Copy> = {
  en: {
    labels: {
      language: "Language",
      backend: "Backend",
      scenario: "Scenario",
      agent: "Agent",
      seed: "Seed",
      steps: "Steps",
      stepBudget: "Step budget",
      recordEpisode: "Record episode",
      stream: "Stream",
      selected: "Selected",
      beamng: "BeamNG",
      activeEpisode: "Active Episode",
      frames: "Frames",
      step: "Step",
      speed: "Speed",
      risk: "Risk",
      terrainRisk: "Terrain Risk",
      localBev: "Local BEV",
      replay: "Replay",
      recentEpisodes: "Recent Episodes"
    },
    status: {
      ok: "ok",
      offline: "offline",
      available: "available",
      unavailable: "unavailable",
      ready: "ready",
      pending: "pending",
      standby: "standby"
    },
    messages: {
      noBackend: "No backend selected",
      noCatalog: "Backend catalogs have not loaded yet.",
      noMetrics: "No metrics yet",
      noMapFrame: "No map frame",
      noEpisodes: "No recorded episodes",
      apiError: "Failed to reach dashboard API.",
      streamBackendOnly: "Streaming run currently targets the local gym_heightmap backend.",
      streamClosed: "Episode stream closed unexpectedly.",
      loadEpisodeFailed: "Could not load episode."
    },
    titles: {
      startStream: "Start streaming run",
      stopStream: "Stop stream",
      refresh: "Refresh catalogs",
      resetReplay: "Reset replay",
      previousFrame: "Previous frame",
      playPause: "Play or pause replay",
      nextFrame: "Next frame"
    },
    metrics: {
      success: "success",
      done: "done",
      steps: "steps",
      total_reward: "total reward",
      distance_to_goal: "distance to goal",
      average_speed: "average speed",
      max_speed: "max speed",
      collision_count: "collisions",
      path_length: "path length",
      average_terrain_risk: "avg terrain risk",
      control_smoothness: "control smoothness"
    }
  },
  zh: {
    labels: {
      language: "语言",
      backend: "后端",
      scenario: "场景",
      agent: "智能体",
      seed: "种子",
      steps: "步数",
      stepBudget: "步数预算",
      recordEpisode: "记录本轮",
      stream: "开始流式运行",
      selected: "当前后端",
      beamng: "BeamNG",
      activeEpisode: "当前 Episode",
      frames: "帧数",
      step: "步",
      speed: "速度",
      risk: "风险",
      terrainRisk: "地形风险",
      localBev: "局部 BEV",
      replay: "回放",
      recentEpisodes: "最近记录"
    },
    status: {
      ok: "正常",
      offline: "离线",
      available: "可用",
      unavailable: "不可用",
      ready: "就绪",
      pending: "待配置",
      standby: "待机"
    },
    messages: {
      noBackend: "尚未选择后端",
      noCatalog: "后端目录尚未加载。",
      noMetrics: "暂无指标",
      noMapFrame: "暂无地图帧",
      noEpisodes: "暂无记录 episode",
      apiError: "无法连接 dashboard API。",
      streamBackendOnly: "当前流式运行仅支持本地 gym_heightmap 后端。",
      streamClosed: "Episode 流意外关闭。",
      loadEpisodeFailed: "无法加载 episode。"
    },
    titles: {
      startStream: "开始流式运行",
      stopStream: "停止流",
      refresh: "刷新目录",
      resetReplay: "重置回放",
      previousFrame: "上一帧",
      playPause: "播放或暂停回放",
      nextFrame: "下一帧"
    },
    metrics: {
      success: "成功",
      done: "完成",
      steps: "步数",
      total_reward: "总奖励",
      distance_to_goal: "距目标",
      average_speed: "平均速度",
      max_speed: "最高速度",
      collision_count: "碰撞次数",
      path_length: "路径长度",
      average_terrain_risk: "平均地形风险",
      control_smoothness: "控制平滑度"
    }
  }
};

const initialLanguage = (): Language => {
  if (typeof window === "undefined") {
    return "zh";
  }
  return window.localStorage.getItem("offroad-sim-language") === "en" ? "en" : "zh";
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
  const [language, setLanguage] = useState<Language>(initialLanguage);
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
  const copy = copyByLanguage[language];

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
      setError(err instanceof Error ? err.message : copy.messages.apiError);
    }
  };

  useEffect(() => {
    void refresh();
    return () => eventSourceRef.current?.close();
  }, []);

  useEffect(() => {
    window.localStorage.setItem("offroad-sim-language", language);
  }, [language]);

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
      setError(copy.messages.streamBackendOnly);
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
      setError(payload?.detail ?? copy.messages.streamClosed);
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
      setError(err instanceof Error ? err.message : copy.messages.loadEpisodeFailed);
    }
  };

  return (
    <main className="app-shell">
      <aside className="control-rail">
        <div className="brand-row">
          <Activity size={22} aria-hidden="true" />
          <div>
            <h1>OffroadSimBench</h1>
            <span className={catalog.health === "ok" ? "status-ok" : "status-bad"}>
              {formatStatus(catalog.health, copy)}
            </span>
          </div>
        </div>

        <div className="language-row" aria-label={copy.labels.language}>
          <Languages size={17} aria-hidden="true" />
          <div className="segmented-control">
            <button
              type="button"
              className={language === "zh" ? "active" : ""}
              onClick={() => setLanguage("zh")}
            >
              中文
            </button>
            <button
              type="button"
              className={language === "en" ? "active" : ""}
              onClick={() => setLanguage("en")}
            >
              EN
            </button>
          </div>
        </div>

        <label>
          {copy.labels.backend}
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
          {copy.labels.scenario}
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
          {copy.labels.agent}
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
            {copy.labels.seed}
            <input
              type="number"
              value={selected.seed}
              onChange={(event) => setSelected({ ...selected, seed: Number(event.target.value) })}
            />
          </label>
          <label>
            {copy.labels.steps}
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
          {copy.labels.stepBudget}
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
          {copy.labels.recordEpisode}
        </label>

        <div className="button-row">
          <button
            type="button"
            className="primary-button"
            onClick={startStream}
            disabled={isStreaming || !canStream}
            title={copy.titles.startStream}
          >
            <Play size={18} aria-hidden="true" />
            <span>{copy.labels.stream}</span>
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={stopStream}
            disabled={!isStreaming}
            title={copy.titles.stopStream}
          >
            <Square size={18} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button" onClick={refresh} title={copy.titles.refresh}>
            <RefreshCw size={18} aria-hidden="true" />
          </button>
        </div>

        {error && <p className="error-text">{error}</p>}

        <BackendStatusPanel backend={selectedBackend} beamng={beamngBackend} copy={copy} />
      </aside>

      <section className="workspace">
        <header className="top-strip">
          <div>
            <span className="eyebrow">{copy.labels.activeEpisode}</span>
            <h2>{liveEpisodeId ?? copy.status.standby}</h2>
            <p>{selectedBackend?.description ?? copy.messages.noBackend}</p>
          </div>
          <div className="top-metrics">
            <MetricBadge label={copy.labels.frames} value={activeFrames.length} />
            <MetricBadge label={copy.labels.step} value={currentFrame?.step_index ?? "-"} />
            <MetricBadge label={copy.labels.speed} value={formatNumber(currentFrame?.observation.vehicle_state.speed)} />
            <MetricBadge label={copy.labels.risk} value={formatNumber(currentFrame?.observation.info.terrain_risk as number)} />
          </div>
        </header>

        <div className="visual-grid">
          <section className="panel map-panel">
            <div className="panel-heading">
              <div>
                <h3>{copy.labels.terrainRisk}</h3>
                <span>{selected.scenario}</span>
              </div>
              <Map size={18} aria-hidden="true" />
            </div>
            <TerrainPanel frames={activeFrames} frameIndex={selectedFrameIndex} copy={copy} />
          </section>

          <section className="panel side-panel">
            <div className="panel-heading">
              <div>
                <h3>{copy.labels.localBev}</h3>
                <span>{copy.labels.step} {currentFrame?.step_index ?? 0}</span>
              </div>
              <Gauge size={18} aria-hidden="true" />
            </div>
            <Heatmap payload={currentFrame?.observation.local_bev ?? null} layer="risk" compact copy={copy} />
            <MetricTable metrics={activeMetrics} copy={copy} />
          </section>
        </div>

        <section className="panel replay-panel">
          <div className="panel-heading">
              <div>
              <h3>{copy.labels.replay}</h3>
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
            copy={copy}
          />
        </section>

        <section className="panel history-panel">
          <div className="panel-heading">
              <div>
              <h3>{copy.labels.recentEpisodes}</h3>
              <span>{catalog.episodes.length}</span>
            </div>
            <Database size={18} aria-hidden="true" />
          </div>
          <EpisodeList episodes={catalog.episodes} onSelect={(episodeId) => void loadEpisode(episodeId)} copy={copy} />
        </section>
      </section>
    </main>
  );
}

function BackendStatusPanel({ backend, beamng, copy }: { backend?: CatalogItem; beamng?: CatalogItem; copy: Copy }) {
  return (
    <div className="runtime-panel">
      <div>
        <span>{copy.labels.selected}</span>
        <strong className={backend?.available === false ? "status-bad" : "status-ok"}>
          {backend?.available === false ? copy.status.unavailable : copy.status.available}
        </strong>
      </div>
      <div>
        <span>{copy.labels.beamng}</span>
        <strong className={beamng?.available ? "status-ok" : "status-warn"}>
          {beamng?.available ? copy.status.ready : copy.status.pending}
        </strong>
      </div>
      <p>{beamng?.message ?? copy.messages.noCatalog}</p>
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

function MetricTable({ metrics, copy }: { metrics: Record<string, unknown>; copy: Copy }) {
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
    return <div className="empty-state">{copy.messages.noMetrics}</div>;
  }
  return (
    <table>
      <tbody>
        {rows.map(([key, value]) => (
          <tr key={key}>
            <th>{copy.metrics[key] ?? key}</th>
            <td>{formatValue(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TerrainPanel({ frames, frameIndex, copy }: { frames: FramePayload[]; frameIndex: number; copy: Copy }) {
  const terrain = latestArray(frames, frameIndex, "terrain_map");
  const trajectory = frames.slice(0, frameIndex + 1).map((frame) => frame.observation.vehicle_state);
  const goal = frames[Math.min(frameIndex, Math.max(frames.length - 1, 0))]?.observation.goal ?? [0, 0];

  return (
    <div className="terrain-stage">
      <Heatmap payload={terrain} layer="risk" copy={copy} />
      <TrajectoryOverlay points={trajectory} goal={goal} />
    </div>
  );
}

function Heatmap({
  payload,
  layer,
  compact = false,
  copy
}: {
  payload?: ArrayLayers | null;
  layer: string;
  compact?: boolean;
  copy: Copy;
}) {
  const matrix = payload?.layers[layer] ?? firstLayer(payload);
  if (!matrix?.length || !matrix[0]?.length) {
    return <div className={compact ? "empty-state compact" : "empty-state"}>{copy.messages.noMapFrame}</div>;
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
  onStepForward,
  copy
}: {
  frameCount: number;
  frameIndex: number;
  isPlaying: boolean;
  onChange: (index: number) => void;
  onReset: () => void;
  onTogglePlay: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  copy: Copy;
}) {
  const disabled = frameCount === 0;
  return (
    <div className="replay-controls">
      <button type="button" className="icon-button light" onClick={onReset} disabled={disabled} title={copy.titles.resetReplay}>
        <RotateCcw size={17} aria-hidden="true" />
      </button>
      <button type="button" className="icon-button light" onClick={onStepBack} disabled={disabled} title={copy.titles.previousFrame}>
        <SkipBack size={17} aria-hidden="true" />
      </button>
      <button type="button" className="icon-button light" onClick={onTogglePlay} disabled={disabled} title={copy.titles.playPause}>
        {isPlaying ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
      </button>
      <button type="button" className="icon-button light" onClick={onStepForward} disabled={disabled} title={copy.titles.nextFrame}>
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
  onSelect,
  copy
}: {
  episodes: EpisodeSummary[];
  onSelect: (episodeId: string) => void;
  copy: Copy;
}) {
  if (!episodes.length) {
    return <div className="empty-state">{copy.messages.noEpisodes}</div>;
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

function formatStatus(status: string, copy: Copy): string {
  if (status === "ok") {
    return copy.status.ok;
  }
  if (status === "offline") {
    return copy.status.offline;
  }
  return status;
}

export default App;
