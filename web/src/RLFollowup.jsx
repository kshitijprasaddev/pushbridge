import React, { useState } from "react";
import { evidenceViewerUrl } from "./evidenceRoutes.mjs";
import "./rl-followup.css";

const VARIANTS = [
  {
    id: "original_20k",
    title: "Original 20k",
    detail: "18 mm training rule",
    className: "baseline",
  },
  {
    id: "longer_original",
    title: "Longer 60k",
    detail: "Same 18 mm training rule",
    className: "longer",
  },
  {
    id: "strict_objective",
    title: "Strict 60k",
    detail: "8 mm throughout the training hold",
    className: "strict",
  },
];

const CASE_STAGES = [
  {
    trainingVersion: "original",
    run: "posttrained_seed2",
    mode: "policy",
    title: "Original 20k policy alone",
    status: "Missed finish",
    fallbackDistance: 111.58353765057865,
    complete: false,
    action: "Replay the missed finish",
  },
  {
    trainingVersion: "longer",
    run: "longer_original_seed2",
    mode: "policy",
    title: "Longer 60k policy alone",
    status: "Much closer, still outside",
    fallbackDistance: 11.167642178031326,
    complete: false,
    action: "Replay after more training",
  },
  {
    trainingVersion: "longer",
    run: "longer_original_seed2_finish",
    mode: "finish",
    title: "Longer 60k + feedback",
    status: "Completed and released",
    fallbackDistance: 2.7670585040705014,
    complete: true,
    action: "Replay the completed finish",
  },
];

function Arrow() {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d="M3 10h13m-5-5 5 5-5 5" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function ScoreCell({ result, seed }) {
  if (!result) return <td>Unavailable</td>;
  return (
    <td
      aria-label={`Model seed ${seed}: ${result.successes} of ${result.episodes} tasks completed`}
    >
      <strong>
        {result.successes}
        <small> / {result.episodes}</small>
      </strong>
      <span className="rf-score-track" aria-hidden="true">
        <i
          style={{ width: `${(result.successes / result.episodes) * 100}%` }}
        />
      </span>
    </td>
  );
}

function RLFollowup({ manifest, onReplay }) {
  const [suiteId, setSuiteId] = useState("paired");
  const data = manifest?.rlFollowup;
  if (!data) return null;
  const suite = data.suites?.[suiteId];
  if (!suite) return null;
  const seeds = data.protocol?.training_seeds || [0, 1, 2];
  const cases = data.protocol?.suites?.[suiteId]?.cases || 20;
  const result = (variant, seed, finish = false) =>
    suite.results?.[`${variant}_seed${seed}${finish ? "_finish" : ""}`];
  const score = (variant, seed, finish) => {
    const value = result(variant, seed, finish);
    return value ? `${value.successes}/${value.episodes}` : "Unavailable";
  };
  return (
    <section className="rf-root" id="rl-followup" aria-labelledby="rf-title">
      <div className="rf-heading">
        <div>
          <span className="rf-eyebrow">
            Follow-up experiment / train for longer
          </span>
          <h2 id="rf-title">
            More practice helped.
            <br />
            It did not solve every finish.
          </h2>
        </div>
        <p>
          Continue each saved compact-arm SAC policy from 20,000 to 60,000
          training decisions. Keep the original training goal in one branch;
          require full containment in the other. Judge both with the same strict
          completion test.
        </p>
      </div>

      <div className="rf-study">
        <div className="rf-study-head">
          <div>
            <span className="rf-eyebrow">Learned policy alone</span>
            <h3>Does the network finish the task?</h3>
          </div>
          <div
            className="rf-suite-control"
            aria-label="Choose evaluation suite"
          >
            <button
              className={suiteId === "paired" ? "selected" : ""}
              aria-pressed={suiteId === "paired"}
              onClick={() => setSuiteId("paired")}
            >
              Original 20 cases
            </button>
            <button
              className={suiteId === "fresh" ? "selected" : ""}
              aria-pressed={suiteId === "fresh"}
              onClick={() => setSuiteId("fresh")}
            >
              Fresh 20 cases
            </button>
          </div>
        </div>
        <p className="rf-suite-description">
          {suiteId === "paired"
            ? "The original cases were inspected before this follow-up. Use them to compare the same task before and after more training."
            : "These new task setups were declared before running the follow-up. They sample the same task and robot, rather than a new embodiment."}
          <span>
            Task-suite seed {suite.seed} · {cases} setups shared by all model
            seeds
          </span>
        </p>
        <div className="rf-table-wrap">
          <table className="rf-table">
            <caption>
              Completed tasks with the learned policy alone, by training version
              and model seed
            </caption>
            <thead>
              <tr>
                <th scope="col">Training version</th>
                {seeds.map((seed) => (
                  <th scope="col" key={seed}>
                    Model seed {seed}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {VARIANTS.map((variant) => (
                <tr key={variant.id} className={variant.className}>
                  <th scope="row">
                    <strong>{variant.title}</strong>
                    <span>{variant.detail}</span>
                  </th>
                  {seeds.map((seed) => (
                    <ScoreCell
                      key={seed}
                      seed={seed}
                      result={result(variant.id, seed)}
                    />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="rf-table-note">
          <span className="rf-rule">
            One test for every row: ≤ 8 mm + one-second hold
          </span>
          <p>
            A model seed is an independently trained network, not a different
            task. Policy-only scores include the benchmark's automatic
            zero-action hold. They do not include the precision controller or
            its tip-release manoeuvre.
          </p>
        </div>
        <div className="rf-reading">
          <span className="rf-eyebrow">What changed</span>
          <p>
            {suiteId === "paired"
              ? "Longer training raises model seed 0 from 6/20 to 18/20, while seed 1 slips from 19/20 to 18/20. The stricter training rule is better for some seeds and worse for others."
              : "With the original training rule, all three longer-trained models complete 19/20 fresh cases. The stricter rule scores 19/20, 18/20 and 18/20, so it does not outperform longer training here."}
          </p>
        </div>
      </div>

      <div className="rf-feedback">
        <div>
          <span className="rf-eyebrow">
            Learned policy + existing precision feedback
          </span>
          <h3>The final controller still matters.</h3>
          <p>
            The finishing code is unchanged. It centres the object, releases the
            tip and verifies the hold. Both 60k versions complete every case in
            both suites when this controller is added.
          </p>
        </div>
        <div className="rf-feedback-scores">
          <div className="rf-feedback-score-head">
            <span>Training version</span>
            {seeds.map((seed) => (
              <span key={seed}>Seed {seed}</span>
            ))}
          </div>
          {VARIANTS.map((variant) => (
            <div
              className={`rf-feedback-score-row ${variant.className}`}
              key={variant.id}
            >
              <span>{variant.title}</span>
              {seeds.map((seed) => (
                <strong key={seed}>{score(variant.id, seed, true)}</strong>
              ))}
            </div>
          ))}
          <p>
            These are hybrid results. Completion includes engineered feedback,
            not learned control alone.
            {suite.results?.feedback_only &&
              ` Feedback alone completes ${suite.results.feedback_only.successes}/${suite.results.feedback_only.episodes} cases in this suite.`}
          </p>
        </div>
      </div>

      <div className="rf-case">
        <div className="rf-case-heading">
          <div>
            <span className="rf-eyebrow">
              Inspect one case / original suite, seed 2, task 04
            </span>
            <h3>Follow the missed finish through the experiment.</h3>
          </div>
          <p>
            Each card loads the same task in the interactive viewer above. The
            distance is measured from the object centre to the target centre.
          </p>
        </div>
        <div className="rf-case-stages">
          {CASE_STAGES.map((stage, i) => {
            const recorded = manifest.runs?.[stage.run]?.trials?.[3];
            const distance =
              recorded?.final_distance_mm ?? stage.fallbackDistance;
            return (
              <div
                className={`rf-case-stage ${stage.complete ? "complete" : "incomplete"}`}
                key={stage.run}
              >
                <span className="rf-case-number">0{i + 1}</span>
                <h4>{stage.title}</h4>
                <strong>
                  {distance.toFixed(3)}
                  <small> mm</small>
                </strong>
                <span className="rf-case-status">{stage.status}</span>
                <button
                  onClick={() =>
                    onReplay?.({
                      trainingVersion: stage.trainingVersion,
                      seed: 2,
                      trial: 3,
                      mode: stage.mode,
                    })
                  }
                  disabled={!onReplay}
                >
                  {stage.action}
                  <Arrow />
                </button>
              </div>
            );
          })}
        </div>
        <p className="rf-case-note">
          More training removes the large miss in this case. The learned policy
          still stops outside full containment; adding the unchanged finisher
          completes it. The original policy also misses with feedback because it
          never reaches the handoff region.
        </p>
      </div>
      <div className="rf-evidence-links">
        <a href={evidenceViewerUrl("/evidence/docs/RL_FOLLOWUP.md")}>
          Read the follow-up method <Arrow />
        </a>
        <a href={evidenceViewerUrl(data.evaluationUrl)}>
          Inspect all follow-up results <Arrow />
        </a>
        <a href={evidenceViewerUrl(data.protocolUrl)}>
          Read the frozen protocol <Arrow />
        </a>
      </div>
      <p className="rf-scope">
        Two small 20-case suites in one simulator task. Repeated model seeds
        share cases. Results include regressions and do not establish
        performance on hardware or new embodiments.
      </p>
    </section>
  );
}

export default React.memo(RLFollowup);
