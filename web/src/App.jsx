import React, { useState, useEffect, useRef, lazy, Suspense } from "react";
const RobotScene = lazy(() => import("./RobotScene.jsx"));
const ResearchPanels = lazy(() => import("./ResearchPanels.jsx"));
const RLFollowup = lazy(() => import("./RLFollowup.jsx"));
import { distanceMM, sampleTrace } from "./kinematics.mjs";
import { evidenceViewerUrl, replaySelection } from "./evidenceRoutes.mjs";

const PHASES = {
  start: ["Initial state", "A single reset places the object and robot."],
  policy: ["Learned policy", "SAC chooses the next tool movement."],
  align: [
    "Align the contact",
    "Feedback positions the narrow tip behind the object.",
  ],
  push: ["Precision push", "Smaller movements close the remaining distance."],
  release: ["Release contact", "The tool retracts so the object can settle."],
  hold: ["Settle", "The tool holds its position."],
  verify: [
    "Verify the finish",
    "The entire object must stay inside for one second.",
  ],
};
const prefixes = {
  direct: "source_direct",
  contact: "source_contact",
  postRL: "posttrained",
};
const EXAMPLES = [
  {
    id: "direct",
    title: "Transfer unchanged",
    note: "An old policy meets a narrower tool.",
    strategy: "direct",
    seed: 1,
    trial: 0,
    mode: "policy",
  },
  {
    id: "contact",
    title: "Adapt the contact",
    note: "Same task. Same weights. New geometry.",
    strategy: "contact",
    seed: 1,
    trial: 0,
    mode: "policy",
  },
  {
    id: "failure",
    title: "Find the RL failure",
    note: "Extra training still leaves a missed finish.",
    strategy: "postRL",
    seed: 2,
    trial: 3,
    mode: "finish",
  },
];
const cache = new Map();
async function loadJSON(url, signal) {
  if (cache.has(url)) return cache.get(url);
  const r = await fetch(url, { signal, cache: "no-cache" });
  if (!r.ok) throw new Error(`Could not load saved evidence (${r.status}).`);
  const value = await r.json();
  cache.set(url, value);
  return value;
}
function Icon({ name, size = 18 }) {
  const paths = {
    play: <path d="m8 5 11 7-11 7z" />,
    pause: (
      <>
        <path d="M8 5v14M16 5v14" />
      </>
    ),
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    diagonal: <path d="M6 18 18 6M6 6h12v12" />,
    reset: (
      <>
        <path d="M4 10a8 8 0 1 1 1 8M4 4v6h6" />
      </>
    ),
    finish: <path d="M5 5v14M5 6h13l-3 4 3 4H5" />,
    expand: <path d="M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    download: <path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4" />,
    check: <path d="m5 12 4 4L19 6" />,
    orbit: (
      <>
        <ellipse cx="12" cy="12" rx="10" ry="5" transform="rotate(-30 12 12)" />
        <circle cx="12" cy="12" r="2" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
function ErrorPlot({ policy, finish, trace, position }) {
  if (!policy || !finish || !trace) return null;
  const series = [policy, finish].map((t) =>
    t.state.map((s) => distanceMM(s, t.goal)),
  );
  const width = 800,
    height = 124,
    maxY = Math.max(45, ...series.flat()),
    maxX = Math.max(...series.map((s) => s.length - 1));
  const x = (i) => 36 + (i / maxX) * 742,
    y = (d) => 96 - (d / maxY) * 74;
  const poly = (values) => values.map((d, i) => `${x(i)},${y(d)}`).join(" ");
  const handoff = finish.phase.findIndex(
    (p) => p !== "policy" && p !== "start",
  );
  return (
    <div className="error-plot">
      <div className="plot-caption">
        <span>DISTANCE TO GOAL</span>
        <div>
          <span className="legend policy">Policy alone</span>
          <span className="legend finish">With feedback</span>
        </div>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Recorded object distance over time. The green line must stay below eight millimetres for full containment."
      >
        <rect
          x="36"
          y={y(8)}
          width="742"
          height={96 - y(8)}
          fill="#d9f97a"
          opacity=".07"
        />
        {[8, maxY].map((d, i) => (
          <g key={i}>
            <line
              x1="36"
              x2="778"
              y1={y(d)}
              y2={y(d)}
              stroke={i ? "#343b3e" : "#a5bb70"}
              strokeDasharray="3 5"
            />
            <text x="0" y={y(d) + 4}>
              {Math.round(d)}
            </text>
          </g>
        ))}
        {handoff >= 0 && (
          <line
            x1={x(handoff)}
            x2={x(handoff)}
            y1="16"
            y2="96"
            stroke="#6c7874"
            strokeDasharray="2 4"
          />
        )}
        <polyline
          points={poly(series[0])}
          fill="none"
          stroke="#878e94"
          strokeWidth="1.8"
        />
        <polyline
          points={poly(series[1])}
          fill="none"
          stroke="#d9f97a"
          strokeWidth="2"
        />
        <line
          x1={x(position)}
          x2={x(position)}
          y1="16"
          y2="96"
          stroke="#f6f8f0"
          opacity=".6"
        />
        <circle
          cx={x(position)}
          cy={y(distanceMM(sampleTrace(trace, position).state, trace.goal))}
          r="3.4"
          fill="#f6f8f0"
        />
        {[0, Math.round(maxX / 2), maxX].map((i) => (
          <text key={i} x={x(i)} y="119" textAnchor="middle">
            {(i * 0.1).toFixed(1)}s
          </text>
        ))}
        <text x="0" y="11">
          mm
        </text>
        <text x="773" y={y(8) - 5} textAnchor="end" className="threshold-text">
          FULL CONTAINMENT
        </text>
      </svg>
    </div>
  );
}
function Containment({ state, goal }) {
  const dx = (state[2] - goal[0]) * 1000,
    dy = (state[3] - goal[1]) * 1000;
  return (
    <svg
      className="containment"
      viewBox="0 0 150 104"
      role="img"
      aria-label="Actual-size object and goal footprint, with the permitted centre region shown by a dotted circle"
    >
      <defs>
        <clipPath id="targetClip">
          <rect width="150" height="104" />
        </clipPath>
      </defs>
      <g clipPath="url(#targetClip)">
        <circle cx="75" cy="52" r="44" fill="#d9f97a" opacity=".12" />
        <circle cx="75" cy="52" r="44" fill="none" stroke="#97ab6d" />
        <circle
          cx="75"
          cy="52"
          r="17.6"
          fill="none"
          stroke="#97ab6d"
          strokeDasharray="2 3"
        />
        <circle
          cx={75 + dx * 2.2}
          cy={52 - dy * 2.2}
          r="26.4"
          fill="#e26b52"
          fillOpacity=".95"
          stroke="#f19679"
        />
        <path d="M71 52h8m-4-4v8" stroke="#f1f4e8" opacity=".5" />
      </g>
    </svg>
  );
}
export default function App() {
  const initial = useRef(replaySelection(location.search)).current;
  const [manifest, setManifest] = useState(null),
    [fatal, setFatal] = useState("");
  const [strategy, setStrategy] = useState(initial?.strategy || "contact"),
    [trainingVersion, setTrainingVersion] = useState(
      initial?.trainingVersion || "original",
    ),
    [seed, setSeed] = useState(initial?.seed ?? 1),
    [trial, setTrial] = useState(initial?.trial ?? 0),
    [mode, setMode] = useState(initial?.mode || "finish");
  const [pair, setPair] = useState(null),
    [loading, setLoading] = useState(true),
    [position, setPosition] = useState(0),
    [playing, setPlaying] = useState(false);
  const [view, setView] = useState("perspective"),
    [showPath, setShowPath] = useState(true),
    [speed, setSpeed] = useState(1),
    [immersive, setImmersive] = useState(false),
    [video, setVideo] = useState(null);
  const stageRef = useRef(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const rlPrefix =
    trainingVersion === "longer"
      ? "longer_original"
      : trainingVersion === "strict"
        ? "strict_objective"
        : "posttrained";
  const selectedPrefix = strategy === "postRL" ? rlPrefix : prefixes[strategy];
  const runID = `${selectedPrefix}_seed${seed}${mode === "finish" ? "_finish" : ""}`;
  const run = manifest?.runs[runID],
    trace = pair?.[mode];
  const finalDistance = trace
    ? distanceMM(trace.state[trace.state.length - 1], trace.goal)
    : 0;
  const finalMeaning = trace
    ? trace.result.success
      ? "The object stays fully inside for the required one-second hold."
      : finalDistance <= 8
        ? "The object ends inside, but the required stationary hold is unfinished."
        : "The object ends outside the region for full containment."
    : "Choose a recorded task to inspect its outcome.";
  const activeExample = EXAMPLES.find(
    (e) =>
      e.strategy === strategy &&
      (strategy !== "postRL" || trainingVersion === "original") &&
      e.seed === seed &&
      e.trial === trial &&
      e.mode === mode,
  );
  const current = trace ? sampleTrace(trace, position) : null;
  const distance = current ? distanceMM(current.state, trace.goal) : 0,
    clearance = 8 - distance;
  const atEnd = trace && position >= trace.state.length - 1;
  const verified = atEnd && trace.result.success;
  const phase = trace
    ? PHASES[trace.phase[current.index]] || PHASES.policy
    : PHASES.start;
  const videoEvidence =
    manifest?.evidence.videos.find(
      (v) =>
        v.run === `${selectedPrefix}_seed${seed}` &&
        v.mode === mode &&
        v.trial === trial,
    ) ||
    manifest?.evidence.videos.find(
      (v) =>
        !v.run &&
        v.strategy === strategy &&
        v.mode === mode &&
        (trainingVersion === "original" || strategy !== "postRL"),
    );
  useEffect(() => {
    loadJSON("/data/manifest.json")
      .then(setManifest)
      .catch((e) => setFatal(e.message));
  }, []);
  useEffect(() => {
    if (!manifest) return;
    const controller = new AbortController();
    setLoading(true);
    setPlaying(false);
    setPair(null);
    const base = `${selectedPrefix}_seed${seed}`;
    Promise.all(
      ["policy", "finish"].map((m) =>
        loadJSON(
          manifest.runs[base + (m === "finish" ? "_finish" : "")].trials[trial]
            .url,
          controller.signal,
        ),
      ),
    )
      .then(([policy, finish]) => {
        setPair({ policy, finish });
        setPosition(
          (modeRef.current === "finish" ? finish : policy).state.length - 1,
        );
        setLoading(false);
        setFatal("");
      })
      .catch((e) => {
        if (e.name !== "AbortError") {
          setFatal(e.message);
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [manifest, selectedPrefix, seed, trial]);
  useEffect(() => {
    if (pair) {
      setPlaying(false);
      setPosition(pair[mode].state.length - 1);
    }
  }, [mode]);
  useEffect(() => {
    if (!playing || !trace) return;
    let last = null,
      frame;
    function advance(now) {
      if (last !== null) {
        const delta = (Math.min(now - last, 100) / 100) * speed;
        setPosition((p) => Math.min(trace.state.length - 1, p + delta));
      }
      last = now;
      frame = requestAnimationFrame(advance);
    }
    frame = requestAnimationFrame(advance);
    return () => cancelAnimationFrame(frame);
  }, [playing, trace, speed]);
  useEffect(() => {
    if (atEnd) setPlaying(false);
  }, [atEnd]);
  useEffect(() => {
    document.body.style.overflow = immersive || video ? "hidden" : "";
    function escape(e) {
      if (e.key === "Escape") {
        setImmersive(false);
        setVideo(null);
      }
    }
    window.addEventListener("keydown", escape);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", escape);
    };
  }, [immersive, video]);
  function replay() {
    if (atEnd) setPosition(0);
    setPlaying((p) => !p);
  }
  function failure() {
    selectExample(EXAMPLES[2]);
  }
  function selectExample(example) {
    if (pair && example.id && activeExample?.id === example.id) {
      setPosition(pair[example.mode].state.length - 1);
    }
    setStrategy(example.strategy);
    setTrainingVersion(example.trainingVersion || "original");
    setSeed(example.seed);
    setTrial(example.trial);
    setMode(example.mode);
    setPlaying(false);
    document.getElementById("lab").scrollIntoView({ behavior: "smooth" });
  }
  const fresh = manifest?.suites.fresh.results;
  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="PushBridge home">
          <svg
            viewBox="0 0 32 32"
            width="32"
            height="32"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M6 25V7h11a6 6 0 0 1 0 12h-5"
              stroke="currentColor"
              strokeWidth="3"
            />
            <circle cx="24" cy="25" r="3" fill="currentColor" />
          </svg>
          <span>
            PushBridge<span className="brand-small">ROBOT LEARNING STUDY</span>
          </span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#lab">Interactive lab</a>
          <a href="#mujoco">MuJoCo footage</a>
          <a href="#research">Research</a>
          <a href="#reproducibility">
            Evidence <Icon name="diagonal" size={13} />
          </a>
          {manifest?.experiment.repository && (
            <a
              href={manifest.experiment.repository}
              target="_blank"
              rel="noreferrer"
            >
              Source <Icon name="diagonal" size={13} />
            </a>
          )}
        </nav>
        <span className="header-note">
          <i /> Independent experiment
        </span>
      </header>
      <main id="top">
        <section className="introduction">
          <div>
            <div className="eyebrow">
              <span className="tiny-square" /> PERSONAL DEMONSTRATIONS / ROBOT
              LEARNING
            </div>
            <h1>
              From five videos.
              <br />
              <span>To robot control.</span>
            </h1>
          </div>
          <div className="intro-aside">
            <p>
              Push the red object fully inside the green circle. The policy
              starts with fingertip demonstrations. A shorter arm and a narrower
              tool test what transfers, and what still needs feedback.
            </p>
            <div className="intro-meta">
              <span>PERSONAL DATA</span>
              <span>MuJoCo + SAC</span>
              <span>REPRODUCIBLE</span>
            </div>
          </div>
        </section>
        <section
          className="guided-comparisons"
          aria-label="Start with three recorded comparisons"
        >
          <div className="guide-caption">
            <span className="tiny-square" /> START WITH THREE COMPARISONS{" "}
            <span>Click a result, then replay it.</span>
          </div>
          <div className="guide-options">
            {EXAMPLES.map((example, i) => (
              <button
                key={example.id}
                aria-pressed={activeExample?.id === example.id}
                disabled={!manifest}
                className={activeExample?.id === example.id ? "active" : ""}
                onClick={() => selectExample(example)}
              >
                <span className="guide-number">0{i + 1}</span>
                <span>
                  <strong>{example.title}</strong>
                  <small>{example.note}</small>
                </span>
                <Icon name="arrow" size={16} />
              </button>
            ))}
          </div>
        </section>
        <section
          className={`lab ${immersive ? "immersive" : ""}`}
          id="lab"
          ref={stageRef}
          aria-label="Interactive robot trajectory lab"
        >
          <div className="lab-heading">
            <div>
              <span className="live-dot" /> ROBOT WORKSPACE{" "}
              <span className="lab-divider">/</span>{" "}
              <span className="muted">THREE-JOINT ARM</span>
            </div>
            <div className="lab-heading-right">
              <span>SAVED PHYSICS, INTERACTIVE VIEW</span>
              <button
                className="icon-button"
                onClick={() => setImmersive((v) => !v)}
                aria-label={
                  immersive ? "Exit expanded view" : "Expand robot viewer"
                }
              >
                <Icon name={immersive ? "close" : "expand"} />
              </button>
            </div>
          </div>
          <div className="lab-grid">
            <div className="viewer-area">
              <div className="scene-wrap">
                <Suspense
                  fallback={
                    <div className="loading-overlay">
                      Preparing the 3D workspace
                    </div>
                  }
                >
                  <RobotScene
                    trace={trace}
                    position={position}
                    view={view}
                    showPath={showPath}
                  />
                </Suspense>
                <div className="camera-options" aria-label="Camera views">
                  {[
                    ["perspective", "Perspective"],
                    ["top", "Top view"],
                    ["detail", "Contact detail"],
                  ].map(([id, label]) => (
                    <button
                      key={id}
                      aria-pressed={view === id}
                      className={view === id ? "active" : ""}
                      onClick={() => setView(id)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="scene-label">
                  <span>PB-03</span>
                  <small>
                    THREE JOINTS / PLANAR PUSHING
                    <br />
                    240 / 170 / 130 mm
                  </small>
                </div>
                <div className="scene-hint">
                  <Icon name="orbit" size={17} />
                  <span>Drag to orbit · Scroll to zoom</span>
                </div>
                {current && (
                  <div className={`finish-stamp ${verified ? "verified" : ""}`}>
                    <span>
                      {verified ? (
                        <Icon name="check" />
                      ) : (
                        <span className="status-ring" />
                      )}
                      {verified
                        ? "TASK COMPLETE"
                        : atEnd
                          ? "TASK INCOMPLETE"
                          : "REPLAYING TASK"}
                    </span>
                    <small>
                      {verified
                        ? "Full containment · 1.0 s hold verified"
                        : atEnd && clearance >= 0
                          ? "Object inside · required hold unfinished"
                          : atEnd
                            ? "Object remains outside the finish region"
                            : phase[0]}
                    </small>
                  </div>
                )}
                {loading && (
                  <div className="loading-overlay">
                    <span className="loading-spinner" />
                    Loading recorded trajectory
                  </div>
                )}
                {fatal && (
                  <div className="load-error" role="alert">
                    {fatal}
                    <button onClick={() => location.reload()}>
                      Reload evidence
                    </button>
                  </div>
                )}
              </div>
              <div className="transport">
                <button
                  className="play-button"
                  disabled={!trace}
                  onClick={replay}
                  aria-label={
                    playing
                      ? "Pause replay"
                      : atEnd
                        ? "Replay task"
                        : "Play replay"
                  }
                >
                  <Icon name={playing ? "pause" : "play"} size={18} />
                  <span>{playing ? "Pause" : "Replay"}</span>
                </button>
                <button
                  className="icon-button"
                  aria-label="Go to initial state"
                  disabled={!trace}
                  onClick={() => {
                    setPlaying(false);
                    setPosition(0);
                  }}
                >
                  <Icon name="reset" size={16} />
                </button>
                <div className="scrubber">
                  <label htmlFor="replay-position" className="sr-only">
                    Replay position
                  </label>
                  <input
                    id="replay-position"
                    type="range"
                    min="0"
                    max={trace ? trace.state.length - 1 : 0}
                    step=".01"
                    value={position}
                    disabled={!trace}
                    style={{
                      "--progress": `${trace ? (position / (trace.state.length - 1)) * 100 : 0}%`,
                    }}
                    onChange={(e) => {
                      setPlaying(false);
                      setPosition(Number(e.target.value));
                    }}
                  />
                </div>
                <span className="clock">
                  {(position * 0.1).toFixed(1)}{" "}
                  <span>
                    /{" "}
                    {trace
                      ? ((trace.state.length - 1) * 0.1).toFixed(1)
                      : "0.0"}{" "}
                    s
                  </span>
                </span>
                <label className="speed-control">
                  <span className="sr-only">Playback speed</span>
                  <select
                    value={speed}
                    onChange={(e) => setSpeed(Number(e.target.value))}
                  >
                    <option value=".5">0.5×</option>
                    <option value="1">1×</option>
                    <option value="2">2×</option>
                  </select>
                </label>
                <button
                  className="icon-button"
                  aria-label="Go to final state"
                  disabled={!trace}
                  onClick={() => {
                    setPlaying(false);
                    setPosition(trace.state.length - 1);
                  }}
                >
                  <Icon name="finish" size={17} />
                </button>
              </div>
              <ErrorPlot
                policy={pair?.policy}
                finish={pair?.finish}
                trace={trace}
                position={position}
              />
            </div>
            <aside className="experiment-controls">
              <div className="control-title">
                <span>01 / HOW THE ROBOT APPROACHES</span>
                <span className="light-label">SAC</span>
              </div>
              <div className="strategy-options">
                {manifest?.strategies.map((s, i) => {
                  const summary =
                    manifest.runs[
                      `${s.id === "postRL" ? rlPrefix : s.prefix}_seed${seed}${mode === "finish" ? "_finish" : ""}`
                    ].summary;
                  return (
                    <button
                      key={s.id}
                      onClick={() => setStrategy(s.id)}
                      aria-pressed={strategy === s.id}
                      className={strategy === s.id ? "selected" : ""}
                    >
                      <span className="strategy-number">0{i + 1}</span>
                      <span className="strategy-name">
                        {s.label}
                        <small>
                          {s.id === "direct"
                            ? "Original policy, different arm"
                            : s.id === "contact"
                              ? "Same weights, adapted contact"
                              : `${trainingVersion === "original" ? "20,000" : "60,000"} target training decisions`}
                        </small>
                      </span>
                      <span className="strategy-score">
                        {summary.successes}
                        <small>/20</small>
                      </span>
                    </button>
                  );
                })}
              </div>
              {strategy === "postRL" && manifest?.rlFollowup && (
                <label className="training-version">
                  RL TRAINING CHECKPOINT
                  <select
                    value={trainingVersion}
                    onChange={(e) => setTrainingVersion(e.target.value)}
                  >
                    <option value="original">
                      Original: 20k / 18 mm training goal
                    </option>
                    <option value="longer">
                      Longer: 60k / 18 mm training goal
                    </option>
                    <option value="strict">
                      Strict: 60k / 8 mm + hold training goal
                    </option>
                  </select>
                  <small>
                    Each 60k model continues the same 20k parent for 40k more
                    decisions. The evaluation rule stays at 8 mm.
                  </small>
                </label>
              )}
              <div className="mode-selector" aria-label="Controller mode">
                <button
                  aria-pressed={mode === "policy"}
                  className={mode === "policy" ? "selected" : ""}
                  onClick={() => setMode("policy")}
                >
                  Policy only
                </button>
                <button
                  aria-pressed={mode === "finish"}
                  className={mode === "finish" ? "selected" : ""}
                  onClick={() => setMode("finish")}
                >
                  + Precision finish
                </button>
              </div>
              <p className="mode-note">
                {mode === "finish"
                  ? "The policy approaches. Within 30 mm, a hand-coded controller makes the final push, releases contact, and checks the hold."
                  : "The learned policy chooses the approach movements. An automatic zero-action hold checks full containment for one second, without the precision controller."}
              </p>
              <div className="run-selectors">
                <label>
                  TRAINING SEED
                  <select
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                  >
                    {[0, 1, 2].map((s) => (
                      <option key={s} value={s}>
                        Seed {s}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  PAIRED TASK
                  <select
                    value={trial}
                    onChange={(e) => setTrial(Number(e.target.value))}
                  >
                    {Array.from({ length: 20 }, (_, i) => (
                      <option key={i} value={i}>
                        Task {String(i + 1).padStart(2, "0")}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <p className="selector-help">
                A seed identifies one trained model. Task 01 to 20 are separate
                starting conditions shared by every strategy. The score above
                counts the whole suite; the scene shows one task.
              </p>
              <div className="control-rule" />
              <div className="control-title">
                <span>02 / DOES THE OBJECT FIT?</span>
                <label className="trace-toggle">
                  <input
                    type="checkbox"
                    checked={showPath}
                    onChange={(e) => setShowPath(e.target.checked)}
                  />
                  Path
                </label>
              </div>
              <div className="telemetry">
                <div className="telemetry-metrics">
                  <span>DISTANCE TO GOAL</span>
                  <strong>
                    {current ? distance.toFixed(1) : "0.0"}
                    <small>mm</small>
                  </strong>
                  <span>SPACE TO THE BOUNDARY</span>
                  <strong className={clearance >= 0 ? "positive" : "negative"}>
                    {clearance >= 0 ? "+" : ""}
                    {current ? clearance.toFixed(1) : "0.0"}
                    <small>mm</small>
                  </strong>
                </div>
                {current && (
                  <Containment state={current.state} goal={trace.goal} />
                )}
              </div>
              <p className="size-note">
                20 mm goal radius. 12 mm object radius.
                <br />
                Its centre must finish within 8 mm.
              </p>
              <div className="phase-readout">
                <span className="phase-icon">
                  {verified ? (
                    <Icon name="check" size={15} />
                  ) : (
                    <span className="live-dot" />
                  )}
                </span>
                <div>
                  <span>{phase[0]}</span>
                  <p>{phase[1]}</p>
                </div>
              </div>
              <div className="run-actions">
                <button
                  disabled={!videoEvidence}
                  onClick={() => {
                    setPlaying(false);
                    setVideo(videoEvidence);
                  }}
                >
                  Watch native MuJoCo clip <Icon name="diagonal" size={14} />
                </button>
                <a href="#mujoco">
                  Browse MuJoCo recordings <Icon name="arrow" size={14} />
                </a>
                {run && (
                  <a href={run.trials[trial].url} download>
                    Download this trace <Icon name="download" size={14} />
                  </a>
                )}
              </div>
              <span className="scope-note">
                Scores: same 20 tasks, selected model seed.
                <br />
                Video:{" "}
                {videoEvidence?.run
                  ? `selected task ${String(trial + 1).padStart(2, "0")}, seed ${seed}`
                  : "reference task 01, seed 1"}
                .
              </span>
            </aside>
          </div>
        </section>
        <section
          className="case-reading"
          aria-live="polite"
          aria-label="Explanation of the selected recorded result"
        >
          <div className="case-title">
            <span className="eyebrow">READ THIS RESULT</span>
            <h3>
              {strategy === "direct"
                ? "Transfer without a contact correction"
                : strategy === "contact"
                  ? "Correct the contact, keep the policy"
                  : "Adapt the policy with more robot training"}
            </h3>
            <p>
              {strategy === "direct"
                ? "The source policy keeps its weights and sees uncorrected contact distances on the narrower tool."
                : strategy === "contact"
                  ? "The adapter rescales tool-to-object distance using their radii. The learned weights are unchanged."
                  : trainingVersion === "original"
                    ? "The adapted policy receives 20,000 SAC training decisions on this arm. It was trained to reach an 18 mm centre error; this evaluation requires 8 mm and a stationary hold."
                    : trainingVersion === "longer"
                      ? "This checkpoint continues the original policy for 40,000 more SAC decisions, reaching 60,000 target decisions. It keeps the original 18 mm training goal, while this evaluation still requires 8 mm."
                      : "This checkpoint continues the same original policy for 40,000 more SAC decisions with an 8 mm goal and a ten-tick hold check during training. The test still counts failures and does not select the best validation checkpoint."}
            </p>
          </div>
          <div className="case-outcome">
            <span>
              RECORDED ENDPOINT / TASK {String(trial + 1).padStart(2, "0")} /
              SEED {seed}
            </span>
            <strong>
              {loading ? "Loading" : `${finalDistance.toFixed(1)} mm`}
              <small>centre error · must be ≤ 8 mm</small>
            </strong>
            <p>
              {loading ? "Loading the saved result." : finalMeaning}{" "}
              {mode === "finish"
                ? "This run includes the hand-coded finish controller."
                : "This run uses the policy alone."}
            </p>
            <button
              disabled={!pair || loading}
              onClick={() => setMode(mode === "finish" ? "policy" : "finish")}
            >
              {mode === "finish"
                ? "Compare policy alone"
                : "Compare with precision feedback"}
              <Icon name="arrow" size={15} />
            </button>
          </div>
        </section>
        <div className="evidence-strip">
          <p>
            <span className="tiny-square" />{" "}
            <strong>Recorded physics, rendered in 3D.</strong> MuJoCo simulated
            the contacts. This viewer replays saved positions and joints; it
            does not run a new experiment.
          </p>
          <button onClick={failure}>
            Inspect a failed transfer <Icon name="arrow" size={16} />
          </button>
        </div>
        <section className="results-summary">
          <div className="summary-lead">
            <div className="eyebrow">A FINISH YOU CAN VERIFY</div>
            <h2>
              Inside means
              <br />
              <span>entirely inside.</span>
            </h2>
            <p>
              Success is a geometric condition followed by a one-second hold. A
              visually close result is still a failed task.
            </p>
          </div>
          <div className="result-table-wrap">
            <div className="table-header">
              <h3>Full containment, by training seed</h3>
              <span>ORIGINAL CHECKPOINTS · 20 PAIRED TASKS</span>
            </div>
            <table className="results-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Policy only</th>
                  <th>+ Precision finish</th>
                </tr>
              </thead>
              <tbody>
                {manifest?.strategies.map((s) => (
                  <tr key={s.id}>
                    <th>{s.label}</th>
                    <td>
                      {[0, 1, 2]
                        .map(
                          (n) =>
                            manifest.runs[`${s.prefix}_seed${n}`].summary
                              .successes,
                        )
                        .join(" / ")}
                      <small>successes out of 20, seeds 0 / 1 / 2</small>
                    </td>
                    <td className="lime-text">
                      {[0, 1, 2]
                        .map(
                          (n) =>
                            manifest.runs[`${s.prefix}_seed${n}_finish`].summary
                              .successes,
                        )
                        .join(" / ")}
                      <small>successes out of 20, seeds 0 / 1 / 2</small>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="table-notes">
              <p>
                The same tasks are repeated across three independently trained
                seeds. The adapter + RL failure is retained and replayable.
              </p>
              <div className="fresh-result">
                <span>
                  FRESH 20 TASKS
                  <br />
                  <small>same distribution, seed 1</small>
                </span>
                <strong>{fresh ? "20 / 20" : "..."}</strong>
                <span>
                  All three hybrid strategies
                  <br />
                  <small>feedback alone: 18 / 20</small>
                </span>
              </div>
            </div>
          </div>
        </section>
        {manifest && (
          <Suspense
            fallback={
              <p className="research-loading">Loading research notebook</p>
            }
          >
            {manifest.rlFollowup && (
              <RLFollowup
                manifest={manifest}
                onReplay={(example) =>
                  selectExample({ ...example, strategy: "postRL" })
                }
              />
            )}
            <ResearchPanels manifest={manifest} />
          </Suspense>
        )}
      </main>
      <footer className="site-footer">
        <a className="brand footer-brand" href="#top">
          PushBridge<span>Study 001 / October 2026</span>
        </a>
        <p>
          A personal research submission for Humanoid.
          <br />
          State-based simulation policies. No robot hardware evaluation.
        </p>
        <a href={evidenceViewerUrl("/evidence/docs/REVIEWER_GUIDE.md")}>
          Reviewer guide <Icon name="diagonal" size={15} />
        </a>
      </footer>
      {video && (
        <div
          className="video-backdrop"
          onClick={(e) => {
            if (e.target === e.currentTarget) setVideo(null);
          }}
        >
          <section
            className="video-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="Original MuJoCo video"
          >
            <div>
              <span>NATIVE MUJOCO EVIDENCE</span>
              <button
                className="icon-button"
                onClick={() => setVideo(null)}
                aria-label="Close physics video"
              >
                <Icon name="close" />
              </button>
            </div>
            <video
              src={video.url}
              poster={video.poster}
              controls
              autoPlay
              playsInline
            />
            <h3>{video.label}</h3>
            <p>
              {video.run
                ? `Selected replay: paired task ${String(video.trial + 1).padStart(2, "0")}, model seed ${video.seed}.`
                : "Reference replay: paired task 01, model seed 1."}{" "}
              This video retains the original physics renderer; the articulated
              lab view above reconstructs saved poses.
            </p>
            <a href={video.url} download>
              Download original video <Icon name="download" size={15} />
            </a>
          </section>
        </div>
      )}
    </>
  );
}
