import React, { useEffect, useRef, useState } from "react";
import "./research.css";

const FALLBACK_CLIPS = [
  { id: "IMG_5670", role: "test", transitions: 111, retained_s: 11.1 },
  { id: "Video", role: "train", transitions: 40, retained_s: 4.0 },
  { id: "Video_1", role: "validation", transitions: 54, retained_s: 5.4 },
  { id: "Video_2", role: "train", transitions: 26, retained_s: 2.6 },
  { id: "Video_3", role: "train", transitions: 58, retained_s: 5.8 },
].map((clip, i) => ({
  ...clip,
  poster: `/media/clip_${String(i).padStart(3, "0")}.jpg`,
  video: `/media/clip_${String(i).padStart(3, "0")}_tracking.mp4`,
}));

const HORIZONS = [
  {
    horizon_steps: 1,
    horizon_seconds: 0.1,
    windows: 111,
    learned_mean_error_mm: 0.894967,
    analytic_mean_error_mm: 1.215461,
    zero_mean_error_mm: 1.644946,
  },
  {
    horizon_steps: 5,
    horizon_seconds: 0.5,
    windows: 22,
    learned_mean_error_mm: 4.058697,
    analytic_mean_error_mm: 5.487588,
    zero_mean_error_mm: 7.664129,
  },
  {
    horizon_steps: 10,
    horizon_seconds: 1,
    windows: 11,
    learned_mean_error_mm: 7.92186,
    analytic_mean_error_mm: 10.296162,
    zero_mean_error_mm: 14.960161,
  },
  {
    horizon_steps: 20,
    horizon_seconds: 2,
    windows: 5,
    learned_mean_error_mm: 13.795649,
    analytic_mean_error_mm: 17.357434,
    zero_mean_error_mm: 28.92396,
  },
  {
    horizon_steps: 40,
    horizon_seconds: 4,
    windows: 2,
    learned_mean_error_mm: 21.815486,
    analytic_mean_error_mm: 32.935515,
    zero_mean_error_mm: 48.511627,
  },
];

const TABS = [
  { id: "transfer", number: "01", label: "Contact transfer" },
  { id: "learning", number: "02", label: "Policy learning" },
  { id: "prediction", number: "03", label: "State prediction" },
  { id: "deployment", number: "04", label: "Actor deployment" },
];

function Arrow({ diagonal = false }) {
  return (
    <svg
      className="rp-arrow"
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
    >
      <path
        d={diagonal ? "M4 16 16 4M4 4h12v12" : "M3 10h13m-5-5 5 5-5 5"}
        stroke="currentColor"
        strokeWidth="1.4"
      />
    </svg>
  );
}

function EvidenceLink({ href, children }) {
  return (
    <a className="rp-text-link" href={href} target="_blank" rel="noreferrer">
      {children}
      <Arrow diagonal />
    </a>
  );
}

function evidenceUrl(manifest, category, pattern, fallback) {
  const items = manifest?.evidence?.[category] || [];
  return (
    items.find((item) => pattern.test(`${item.label || ""} ${item.url || ""}`))
      ?.url || fallback
  );
}

function ClipDialog({ clip, onClose }) {
  const closeButton = useRef(null);
  useEffect(() => {
    const previous = document.activeElement;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButton.current?.focus();
    const keydown = (event) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") {
        const controls = [
          ...document.querySelectorAll(
            ".rp-clip-dialog button, .rp-clip-dialog video, .rp-clip-dialog a",
          ),
        ];
        const first = controls[0],
          last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        }
        if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener("keydown", keydown);
      previous?.focus?.();
    };
  }, [onClose]);
  return (
    <div
      className="rp-dialog-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="rp-clip-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rp-clip-title"
      >
        <div className="rp-dialog-head">
          <div>
            <span className="rp-eyebrow">
              Personal recording / tracking overlay
            </span>
            <h3 id="rp-clip-title">{clip.id}.mov</h3>
          </div>
          <button
            ref={closeButton}
            onClick={onClose}
            aria-label="Close recording"
          >
            ✕
          </button>
        </div>
        <video
          controls
          playsInline
          preload="metadata"
          poster={clip.poster}
          src={clip.video}
          tabIndex={0}
        />
        <div className="rp-dialog-foot">
          <span>
            <i className={`rp-split-dot ${clip.role}`} />
            {clip.role === "test"
              ? "Held-out test recording"
              : clip.role === "train"
                ? "Training recording"
                : "Validation recording"}
          </span>
          <span>{clip.transitions} usable transitions</span>
        </div>
        <p>
          Red marks the object; blue marks the fingertip. Coordinates use the
          drawn target as a reference. The video pixels are recorded footage.
        </p>
      </div>
    </div>
  );
}

function RecordingStrip({ manifest }) {
  const [selectedClip, setSelectedClip] = useState(null);
  const clips = manifest?.clips?.length ? manifest.clips : FALLBACK_CLIPS;
  return (
    <div className="rp-data-block" id="data">
      <div className="rp-data-intro">
        <div>
          <span className="rp-eyebrow">Dataset / recorded at home</span>
          <h3>
            Five clips.
            <br />A complete provenance trail.
          </h3>
        </div>
        <p>
          Push a red cap into a circle with a fingertip. Track the contact,
          retain continuous motion, and split entire recordings before learning.
        </p>
      </div>
      <div className="rp-recording-strip">
        {clips.map((clip, i) => {
          const role = clip.role === "val" ? "validation" : clip.role;
          return (
            <button
              className="rp-recording"
              onClick={() => setSelectedClip({ ...clip, role })}
              key={clip.id}
              aria-label={`Watch ${clip.id}, ${role} recording`}
            >
              <div className="rp-recording-image">
                <img
                  src={clip.poster}
                  alt={`Recorded fingertip pushing the red object, clip ${i + 1}`}
                  loading="lazy"
                />
                <span className="rp-clip-number">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className={`rp-split-badge ${role}`}>
                  {role === "test"
                    ? "Held-out test"
                    : role === "validation"
                      ? "Validation"
                      : "Train"}
                </span>
                <span className="rp-clip-play">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="m9 5 11 7-11 7z" fill="currentColor" />
                  </svg>
                </span>
              </div>
              <div className="rp-recording-caption">
                <span>{clip.id}</span>
                <span>{clip.transitions} transitions</span>
              </div>
            </button>
          );
        })}
      </div>
      <div className="rp-data-foot">
        <span>
          <strong>289</strong> usable transitions{" "}
          <span className="rp-separator">/</span> <strong>195</strong>{" "}
          moving-object transitions
        </span>
        <p>
          3 train · 1 validation · 1 test. Whole clips stay in one split. The 20
          mm target is a chosen simulation scale, not a physical measurement of
          the recordings.
        </p>
      </div>
      {selectedClip && (
        <ClipDialog clip={selectedClip} onClose={() => setSelectedClip(null)} />
      )}
    </div>
  );
}

function RobotSchematic() {
  return (
    <svg
      className="rp-robot-schematic"
      viewBox="0 0 760 285"
      role="img"
      aria-labelledby="rp-robot-title rp-robot-description"
    >
      <title id="rp-robot-title">
        Source arm and compact target arm geometry
      </title>
      <desc id="rp-robot-description">
        The source has two links of 320 and 300 millimetres and an 8 millimetre
        radius tip. The target has three links of 240, 170 and 130 millimetres
        and a 3 millimetre radius tip. These are schematic poses, not simulation
        frames.
      </desc>
      <defs>
        <pattern
          id="rp-cad-grid"
          width="24"
          height="24"
          patternUnits="userSpaceOnUse"
        >
          <path d="M24 0H0V24" fill="none" stroke="#31383c" strokeWidth=".6" />
        </pattern>
      </defs>
      <rect width="760" height="285" fill="url(#rp-cad-grid)" />
      <line x1="380" y1="0" x2="380" y2="285" stroke="#394145" />
      <g fontFamily="monospace" fontSize="10" fill="#8e999f">
        <text x="22" y="28">
          SOURCE / 2R
        </text>
        <text x="402" y="28">
          TARGET / COMPACT 3R
        </text>
        <text x="22" y="264">
          SCHEMATIC GEOMETRY
        </text>
        <text x="402" y="264">
          JOINT LIMITS + 0.25 RAD/S SLEW LIMIT
        </text>
      </g>
      <g stroke="#647078" strokeWidth="1" fill="none" strokeDasharray="3 4">
        <line x1="66" y1="222" x2="331" y2="222" />
        <line x1="66" y1="72" x2="66" y2="237" />
        <line x1="427" y1="222" x2="731" y2="222" />
        <line x1="427" y1="72" x2="427" y2="237" />
      </g>
      <g fill="none" strokeLinecap="round">
        <path d="M66 222 180 96 310 169" stroke="#2a3035" strokeWidth="25" />
        <path d="M66 222 180 96 310 169" stroke="#8f9aa0" strokeWidth="14" />
        <path
          d="M427 222 516 127 592 175 666 116"
          stroke="#2a3035"
          strokeWidth="25"
        />
        <path
          d="M427 222 516 127 592 175 666 116"
          stroke="#d9f97a"
          strokeWidth="14"
        />
      </g>
      {[
        [66, 222],
        [180, 96],
        [427, 222],
        [516, 127],
        [592, 175],
      ].map(([cx, cy], i) => (
        <g key={i}>
          <circle cx={cx} cy={cy} r="17" fill="#182025" stroke="#7c8a91" />
          <circle cx={cx} cy={cy} r="8" fill="#101316" stroke="#7c8a91" />
          <circle cx={cx} cy={cy} r="2.5" fill="#c5ced1" />
        </g>
      ))}
      <circle cx="310" cy="169" r="8" fill="#78bed1" />
      <circle cx="666" cy="116" r="3" fill="#78bed1" />
      <g fontFamily="monospace" fontSize="11" fill="#dbe3e5">
        <text x="97" y="136" transform="rotate(-48 97 136)">
          320 mm
        </text>
        <text x="227" y="113" transform="rotate(29 227 113)">
          300 mm
        </text>
        <text x="445" y="153" transform="rotate(-47 445 153)">
          240 mm
        </text>
        <text x="545" y="139" transform="rotate(32 545 139)">
          170 mm
        </text>
        <text x="619" y="137" transform="rotate(-39 619 137)">
          130 mm
        </text>
      </g>
      <path d="M310 181v27h38M666 129v63h55" stroke="#78bed1" fill="none" />
      <g fontFamily="monospace" fontSize="11" fill="#78bed1">
        <text x="302" y="226">
          TIP R 8 mm
        </text>
        <text x="640" y="214">
          TIP R 3 mm
        </text>
      </g>
    </svg>
  );
}

function TransferPanel({ manifest }) {
  return (
    <div
      className="rp-study-panel"
      role="tabpanel"
      id="rp-panel-transfer"
      aria-labelledby="rp-tab-transfer"
    >
      <div className="rp-study-copy">
        <span className="rp-eyebrow">Embodiment retargeting</span>
        <h3>
          Different joints.
          <br />
          Different contact geometry.
        </h3>
        <p>
          The target arm is shorter, has a third joint, and pushes through a
          narrower tip. The same Cartesian command produces different reachable
          motion and contact clearance.
        </p>
        <p>
          The adapter expresses tool position relative to the object, corrects
          for the sum of their radii, then transforms the policy action back
          into target coordinates.
        </p>
        <EvidenceLink
          href={evidenceUrl(
            manifest,
            "docs",
            /EXPERIMENTS/i,
            "/evidence/docs/EXPERIMENTS.md",
          )}
        >
          Read the transfer experiment
        </EvidenceLink>
      </div>
      <div className="rp-study-figure">
        <RobotSchematic />
        <div className="rp-contact-equation">
          <span>Contact-space scale</span>
          <div>
            <span>(12 + 8)</span>
            <span className="rp-equation-divider">/</span>
            <span>(12 + 3)</span>
            <span className="rp-equation-equals">=</span>
            <strong>4 / 3</strong>
          </div>
          <p>
            Object radius stays at 12 mm. The frozen actor keeps its original
            weights.
          </p>
        </div>
        <div className="rp-figure-note">
          <span className="rp-note-dot" />
          Compact target trials use bounded inverse kinematics and physical
          MuJoCo contacts.
        </div>
      </div>
    </div>
  );
}

function SeedBars() {
  const rows = [
    { label: "Scratch SAC", values: [13, 9, 20], className: "scratch" },
    {
      label: "Personal init + SAC",
      values: [20, 20, 20],
      className: "personal",
    },
  ];
  return (
    <div className="rp-seed-chart">
      <div className="rp-figure-head">
        <span>Source-arm evaluation</span>
        <span>20 tasks · original 18 mm rule</span>
      </div>
      <div className="rp-seed-head">
        <span>Successful tasks</span>
        <span>Seed 0</span>
        <span>Seed 1</span>
        <span>Seed 2</span>
      </div>
      {rows.map((row) => (
        <div className={`rp-seed-row ${row.className}`} key={row.label}>
          <strong>{row.label}</strong>
          {row.values.map((value, i) => (
            <div className="rp-seed-cell" key={i}>
              <span className="rp-seed-bar">
                <i style={{ width: `${(value / 20) * 100}%` }} />
              </span>
              <span>
                {value}
                <small> / 20</small>
              </span>
            </div>
          ))}
        </div>
      ))}
      <div className="rp-seed-chart-foot">
        Same 20 cases across training seeds. Classical feedback and MPC also
        solve 20/20 under this original criterion.
      </div>
    </div>
  );
}

function LearningPanel({ manifest }) {
  return (
    <div
      className="rp-study-panel"
      role="tabpanel"
      id="rp-panel-learning"
      aria-labelledby="rp-tab-learning"
    >
      <div className="rp-study-copy">
        <span className="rp-eyebrow">
          Bootstrap, then learn through contact
        </span>
        <h3>
          Personal motion
          <br />
          initializes the actor.
        </h3>
        <p>
          123 filtered training transitions, augmented by in-plane rotations,
          initialize the actor's mean action. SAC then learns from 40,000
          simulator interactions per training seed.
        </p>
        <p>
          The compact robot receives the contact adapter and a separate
          20,000-step target RL run. The original checkpoints stay frozen for
          the strict completion study above.
        </p>
        <EvidenceLink
          href={evidenceUrl(
            manifest,
            "docs",
            /EXPERIMENTS/i,
            "/evidence/docs/EXPERIMENTS.md",
          )}
        >
          Inspect training budgets and baselines
        </EvidenceLink>
      </div>
      <div className="rp-study-figure">
        <SeedBars />
        <div className="rp-negative-result">
          <span className="rp-eyebrow">
            Additional experiment / personal post-training
          </span>
          <h4>Some useful updates also hurt.</h4>
          <p>
            Adding 200 supervised actor updates after scratch SAC changes source
            successes from <strong>13 → 20</strong>, <strong>9 → 5</strong>, and{" "}
            <strong>20 → 19</strong> across the three seeds. Human action labels
            can conflict with the robot's learned behavior.
          </p>
          <span className="rp-quiet-note">
            Measured at the original 18 mm threshold. This result is preserved
            alongside the improvements.
          </span>
        </div>
      </div>
    </div>
  );
}

function PredictionChart({ horizons, selected }) {
  const w = 650,
    h = 246,
    left = 48,
    right = 20,
    top = 24,
    bottom = 34;
  const x = (time) => left + (time / 4) * (w - left - right);
  const y = (error) => h - bottom - (error / 50) * (h - top - bottom);
  const path = (field) =>
    horizons
      .map(
        (point, i) =>
          `${i ? "L" : "M"}${x(point.horizon_seconds)},${y(point[field])}`,
      )
      .join(" ");
  return (
    <svg
      className="rp-prediction-chart"
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-labelledby="rp-prediction-title"
    >
      <title id="rp-prediction-title">
        Held-out object state prediction error grows with open-loop horizon. The
        learned model has lower mean error than the fixed analytic and
        zero-motion baselines at all five measured horizons.
      </title>
      {[0, 10, 20, 30, 40, 50].map((tick) => (
        <g key={tick}>
          <line
            x1={left}
            x2={w - right}
            y1={y(tick)}
            y2={y(tick)}
            stroke="#32393d"
            strokeWidth=".6"
          />
          <text
            x={left - 12}
            y={y(tick) + 4}
            textAnchor="end"
            fill="#89969d"
            fontSize="10"
            fontFamily="monospace"
          >
            {tick}
          </text>
        </g>
      ))}
      {[0, 1, 2, 3, 4].map((tick) => (
        <text
          key={tick}
          x={x(tick)}
          y={h - 10}
          textAnchor="middle"
          fill="#89969d"
          fontSize="10"
          fontFamily="monospace"
        >
          {tick}s
        </text>
      ))}
      <text x={left} y="12" fill="#89969d" fontSize="10" fontFamily="monospace">
        MEAN OBJECT ERROR / mm
      </text>
      <path
        d={`${path("learned_mean_error_mm")} L${x(4)},${y(0)} L${x(0.1)},${y(0)} Z`}
        fill="#d9f97a"
        fillOpacity=".045"
      />
      <path
        d={path("zero_mean_error_mm")}
        fill="none"
        stroke="#7b858b"
        strokeWidth="1.5"
        strokeDasharray="3 4"
      />
      <path
        d={path("analytic_mean_error_mm")}
        fill="none"
        stroke="#78bed1"
        strokeWidth="1.5"
        strokeDasharray="5 3"
      />
      <path
        d={path("learned_mean_error_mm")}
        fill="none"
        stroke="#d9f97a"
        strokeWidth="2"
      />
      <line
        x1={x(selected.horizon_seconds)}
        x2={x(selected.horizon_seconds)}
        y1={top}
        y2={h - bottom}
        stroke="#d9f97a"
        strokeOpacity=".28"
      />
      {horizons.map((point) => (
        <circle
          key={point.horizon_steps}
          cx={x(point.horizon_seconds)}
          cy={y(point.learned_mean_error_mm)}
          r={point.horizon_steps === selected.horizon_steps ? 5 : 3}
          fill="#101316"
          stroke="#d9f97a"
          strokeWidth="1.5"
        />
      ))}
    </svg>
  );
}

function PredictionPanel({ manifest }) {
  const horizons = manifest?.world?.horizons?.length
    ? manifest.world.horizons
    : HORIZONS;
  const [selectedIndex, setSelectedIndex] = useState(2);
  const selected = horizons[selectedIndex] || horizons[0];
  const video = manifest?.world?.video || "/evidence/prediction_overlay.mp4";
  return (
    <div
      className="rp-study-panel"
      role="tabpanel"
      id="rp-panel-prediction"
      aria-labelledby="rp-tab-prediction"
    >
      <div className="rp-study-copy">
        <span className="rp-eyebrow">World modelling / measured states</span>
        <h3>
          Predict what
          <br />
          the object does next.
        </h3>
        <p>
          A three-member model ensemble learns object displacement from personal
          contact tracks. Future tool actions are supplied; object state rolls
          forward without resetting inside each window.
        </p>
        <p>
          This is action-conditioned state prediction, projected onto held-out
          footage. The frames come from the recording. The model does not
          generate video pixels.
        </p>
        <EvidenceLink href={video}>
          Watch prediction on recorded footage
        </EvidenceLink>
      </div>
      <div className="rp-study-figure">
        <div className="rp-figure-head">
          <span>Held-out recording / IMG_5670</span>
          <span>Open-loop prediction</span>
        </div>
        <PredictionChart horizons={horizons} selected={selected} />
        <div className="rp-chart-legend">
          <span>
            <i className="learned" />
            Learned model
          </span>
          <span>
            <i className="analytic" />
            Fixed contact model
          </span>
          <span>
            <i className="zero" />
            Zero motion
          </span>
        </div>
        <div className="rp-horizon-control">
          <div className="rp-horizon-buttons" aria-label="Prediction horizon">
            {horizons.map((point, i) => (
              <button
                key={point.horizon_steps}
                aria-pressed={selectedIndex === i}
                className={selectedIndex === i ? "selected" : ""}
                onClick={() => setSelectedIndex(i)}
              >
                {point.horizon_seconds}s
              </button>
            ))}
          </div>
          <div className="rp-horizon-stat">
            <strong>
              {selected.learned_mean_error_mm.toFixed(2)}
              <small> mm</small>
            </strong>
            <span>
              {selected.windows} held-out windows at {selected.horizon_steps}{" "}
              steps
            </span>
          </div>
        </div>
        <div className="rp-figure-note">
          <span className="rp-note-dot" />
          One test recording; longer horizons have fewer windows. Distances use
          the chosen simulator scale.
        </div>
      </div>
    </div>
  );
}

function DeploymentPanel({ manifest }) {
  const latency = manifest?.speed?.latency;
  const reference = latency?.sb3_predict?.p50_ms || 0.1747655;
  const direct = latency?.torch_direct?.p50_ms || 0.0445635;
  return (
    <div
      className="rp-study-panel"
      role="tabpanel"
      id="rp-panel-deployment"
      aria-labelledby="rp-tab-deployment"
    >
      <div className="rp-study-copy">
        <span className="rp-eyebrow">Inference engineering</span>
        <h3>
          The same actor.
          <br />
          Less inference overhead.
        </h3>
        <p>
          Export the deterministic mean actor and remove training and
          distribution wrappers. Direct Torch inference uses the same float32
          kernels and produces identical actions and states in all 20 compared
          rollouts.
        </p>
        <p>
          Timing covers observation encoding, actor inference, and action
          clipping. It excludes physics and rendering. These are local CPU
          measurements, not robot hardware latency.
        </p>
        <EvidenceLink
          href={evidenceUrl(
            manifest,
            "downloads",
            /actor.*npz|actor.*export/i,
            "/evidence/actor.npz",
          )}
        >
          Download the exported actor
        </EvidenceLink>
      </div>
      <div className="rp-study-figure">
        <div className="rp-speed-top">
          <div>
            <span className="rp-eyebrow">Median actor latency</span>
            <strong>
              {(reference / direct).toFixed(2)}
              <span>×</span>
            </strong>
            <p>faster with direct Torch</p>
          </div>
          <div className="rp-identical-stamp">
            <svg viewBox="0 0 32 32" aria-hidden="true">
              <path
                d="m7 16 6 6 13-14"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              />
            </svg>
            <span>
              20 / 20 traces
              <br />
              <strong>Bit-identical</strong>
            </span>
          </div>
        </div>
        <div className="rp-latency-row">
          <span>SB3 prediction wrapper</span>
          <div>
            <i style={{ width: "100%" }} />
          </div>
          <strong>
            {reference.toFixed(3)}
            <small> ms</small>
          </strong>
        </div>
        <div className="rp-latency-row direct">
          <span>Direct Torch actor</span>
          <div>
            <i style={{ width: `${(direct / reference) * 100}%` }} />
          </div>
          <strong>
            {direct.toFixed(3)}
            <small> ms</small>
          </strong>
        </div>
        <div className="rp-actor-architecture">
          <span>4 inputs</span>
          <i />
          <span>64 ReLU</span>
          <i />
          <span>64 ReLU</span>
          <i />
          <span>2 actions</span>
        </div>
        <div className="rp-figure-note">
          <span className="rp-note-dot" />
          4,610 parameters · 18.4 kB export · 3,000 timed calls per method after
          warm-up
        </div>
      </div>
    </div>
  );
}

function Reproducibility({ manifest }) {
  const [copied, setCopied] = useState(false);
  const command =
    "python -m pip install -e '.[dev,analysis,rl]'\npython -m pytest -q\npython scripts/audit_submission.py";
  const fallback = [
    {
      label: "Two-minute reviewer guide",
      url: "/evidence/docs/REVIEWER_GUIDE.md",
      format: "MD",
      description: "A short path through the question, experiments, and scope.",
    },
    {
      label: "Experiment methodology",
      url: "/evidence/docs/EXPERIMENTS.md",
      format: "MD",
      description:
        "Data splits, budgets, controller baselines, and reproduction.",
    },
    {
      label: "Strict completion protocol",
      url: "/evidence/docs/PRECISION.md",
      format: "MD",
      description: "Containment, release, stationary hold, and attribution.",
    },
    {
      label: "Every completion result",
      url: "/evidence/results/precision_finish/evaluation.json",
      format: "JSON",
      description: "All 520 saved trials, including the remaining failures.",
    },
    {
      label: "Evidence audit",
      url: "/evidence/results/submission_audit.json",
      format: "JSON",
      description: "Hashes, trace consistency, action limits, and hold checks.",
    },
  ];
  const docs = manifest?.evidence?.docs || [];
  const results = manifest?.evidence?.results || [];
  const downloads = manifest?.evidence?.downloads || [];
  const files = fallback.map((file) => ({
    ...file,
    url:
      [...docs, ...results].find((item) =>
        item.url?.endsWith(file.url.split("/").pop()),
      )?.url || file.url,
  }));
  const actor = downloads.find((item) =>
    /actor|checkpoint|policy/i.test(`${item.label} ${item.url}`),
  );
  if (actor)
    files.push({
      ...actor,
      format: actor.format || (actor.url.endsWith(".npz") ? "NPZ" : "ZIP"),
      description:
        "Included policy weights, ready for inspection and reproduction.",
    });
  return (
    <section className="rp-reproduce" id="reproducibility">
      <div className="rp-reproduce-intro">
        <span className="rp-eyebrow">Review the evidence</span>
        <h2>
          Inspectable.
          <br />
          Reproducible.
        </h2>
        <p>
          Every reported result has a protocol, a saved checkpoint, or a
          recorded trace behind it. Start with the guide; follow a result back
          to its source.
        </p>
        <div className="rp-scope-note">
          <span className="rp-eyebrow">Scope of this submission</span>
          <p>
            State-based policies in planar MuJoCo tasks. Exact simulator
            positions are inputs. The terminal precision controller is
            hand-coded. Hardware transfer and image or language-conditioned
            control are future work.
          </p>
        </div>
      </div>
      <div className="rp-evidence-files">
        {files.map((file) => (
          <a
            className="rp-evidence-file"
            href={file.url}
            key={file.url}
            target="_blank"
            rel="noreferrer"
          >
            <span className="rp-file-format">{file.format}</span>
            <div>
              <strong>{file.label}</strong>
              <p>{file.description}</p>
            </div>
            <Arrow diagonal />
          </a>
        ))}
      </div>
      <details className="rp-checkpoint-library">
        <summary>
          Policy and model library{" "}
          <span>{downloads.length} verified files</span>
        </summary>
        <div>
          {downloads.map((item) => (
            <a key={item.url} href={item.url} download>
              <span>
                {item.label}
                <small>
                  {item.format} · {(item.bytes / 1024).toFixed(1)} kB
                </small>
              </span>
              <Arrow diagonal />
            </a>
          ))}
        </div>
      </details>
      <div className="rp-reproduce-command">
        <div>
          <span className="rp-eyebrow">Run from the repository root</span>
          <h3>Verify the submission locally.</h3>
          <p>
            The audit reads the included artifacts. No retraining is required.
          </p>
        </div>
        <div className="rp-command-block">
          <button
            aria-label="Copy verification commands"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(command);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              } catch {
                setCopied(false);
              }
            }}
          >
            {copied ? "Copied" : "Copy"}
            <Arrow />
          </button>
          <pre>
            <code>{command}</code>
          </pre>
        </div>
      </div>
    </section>
  );
}

function ResearchPanels({ manifest }) {
  const [activeTab, setActiveTab] = useState("transfer");
  const panel = {
    transfer: TransferPanel,
    learning: LearningPanel,
    prediction: PredictionPanel,
    deployment: DeploymentPanel,
  };
  const Panel = panel[activeTab];
  return (
    <div className="rp-root">
      <section className="rp-research" id="research">
        <div className="rp-section-heading">
          <div>
            <span className="rp-eyebrow">Research notebook / 01</span>
            <h2>
              A small dataset.
              <br />A traceable learning system.
            </h2>
          </div>
          <p>
            The graphics above make the task visible. The work below explains
            where the policy comes from, how it transfers, and what the evidence
            supports.
          </p>
        </div>
        <RecordingStrip manifest={manifest} />
        <div className="rp-methodology">
          <div
            className="rp-tab-list"
            role="tablist"
            aria-label="Research experiments"
          >
            {TABS.map((tab) => (
              <button
                key={tab.id}
                id={`rp-tab-${tab.id}`}
                role="tab"
                aria-selected={activeTab === tab.id}
                aria-controls={`rp-panel-${tab.id}`}
                tabIndex={activeTab === tab.id ? 0 : -1}
                className={activeTab === tab.id ? "active" : ""}
                onClick={() => setActiveTab(tab.id)}
                onKeyDown={(event) => {
                  if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
                    event.preventDefault();
                    const index = TABS.findIndex((item) => item.id === tab.id);
                    const next =
                      TABS[
                        (index +
                          (event.key === "ArrowRight" ? 1 : -1) +
                          TABS.length) %
                          TABS.length
                      ];
                    setActiveTab(next.id);
                    document.getElementById(`rp-tab-${next.id}`)?.focus();
                  }
                }}
              >
                <span>{tab.number}</span>
                {tab.label}
                <Arrow />
              </button>
            ))}
          </div>
          <Panel manifest={manifest} />
        </div>
      </section>
      <Reproducibility manifest={manifest} />
    </div>
  );
}

export default React.memo(ResearchPanels);
