import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import {
  forwardKinematics,
  worldPosition,
  sampleTrace,
  distanceMM,
} from "../src/kinematics.mjs";
const publicRoot = resolve("public");
const json = (path) =>
  JSON.parse(
    readFileSync(resolve(publicRoot, path.replace(/^\//, "")), "utf8"),
  );
const manifest = json("data/manifest.json");

test("world-to-renderer transform preserves handedness and physical lengths", () => {
  const points = forwardKinematics([Math.PI / 2, 0, 0]);
  assert.ok(Math.abs(points.at(-1)[0]) < 1e-12);
  assert.ok(Math.abs(points.at(-1)[1] - 0.54) < 1e-12);
  assert.deepEqual(worldPosition(0, 0.105, 0.01), [0.25, 0.01, -0]);
  const bent = forwardKinematics([0.4, 1.1, -0.5]);
  [0.24, 0.17, 0.13].forEach((length, i) =>
    assert.ok(
      Math.abs(
        Math.hypot(bent[i + 1][0] - bent[i][0], bent[i + 1][1] - bent[i][1]) -
          length,
      ) < 1e-12,
    ),
  );
});

test("all visible replays retain endpoint, physical geometry, and verified holds", () => {
  let count = 0,
    maximumFKResidual = 0;
  for (const run of Object.values(manifest.runs)) {
    let successes = 0;
    for (const entry of run.trials) {
      const trace = json(entry.url);
      count++;
      assert.equal(trace.state.length, trace.joint.length);
      assert.equal(trace.state.length, trace.phase.length);
      assert.equal(trace.action.length + 1, trace.state.length);
      assert.ok(
        Math.abs(
          distanceMM(trace.state.at(-1), trace.goal) - entry.final_distance_mm,
        ) < 1e-8,
      );
      trace.joint.forEach((q, i) => {
        const fk = forwardKinematics(q).at(-1);
        const target = [trace.state[i][0] + 0.25, trace.state[i][1] - 0.105];
        maximumFKResidual = Math.max(
          maximumFKResidual,
          Math.hypot(fk[0] - target[0], fk[1] - target[1]) * 1000,
        );
      });
      assert.deepEqual(
        sampleTrace(trace, trace.state.length - 1).state,
        trace.state.at(-1),
      );
      if (entry.success) {
        successes++;
        assert.ok(
          trace.state.slice(-10).every((s) => distanceMM(s, trace.goal) <= 8),
        );
        assert.ok(
          trace.action.slice(-10).every((a) => a.every((v) => v === 0)),
        );
        assert.ok(trace.phase.slice(-10).every((p) => p === "verify"));
      }
    }
    assert.equal(successes, run.summary.successes);
  }
  const originalCount = Object.entries(manifest.runs)
    .filter(([name]) => !/^(longer_original|strict_objective)_/.test(name))
    .reduce((sum, [, run]) => sum + run.trials.length, 0);
  assert.equal(originalCount, 360);
  assert.equal(count, manifest.rlFollowup ? 600 : 360);
  if (manifest.rlFollowup)
    assert.equal(
      count - originalCount,
      manifest.rlFollowup.pairedReplaysExported,
    );
  assert.ok(
    maximumFKResidual < 0.161,
    `Recorded MuJoCo site/joint timing residual ${maximumFKResidual} mm exceeds measured bound`,
  );
});

test("longer training keeps the original failure and separates learned from feedback completion", () => {
  if (!manifest.rlFollowup) return;
  const old = manifest.runs.posttrained_seed2.trials[3];
  const longer = manifest.runs.longer_original_seed2.trials[3];
  const finished = manifest.runs.longer_original_seed2_finish.trials[3];
  assert.equal(old.success, false);
  assert.equal(longer.success, false);
  assert.equal(finished.success, true);
  assert.ok(old.final_distance_mm > 100);
  assert.ok(longer.final_distance_mm > 8 && longer.final_distance_mm < 12);
  assert.ok(finished.final_distance_mm < 8);
  for (const seed of [0, 1, 2]) {
    assert.equal(
      manifest.rlFollowup.suites.paired.results[`original_20k_seed${seed}`]
        .successes,
      manifest.runs[`posttrained_seed${seed}`].summary.successes,
    );
    for (const variant of ["longer_original", "strict_objective"]) {
      assert.equal(
        manifest.rlFollowup.suites.paired.results[`${variant}_seed${seed}`]
          .successes,
        manifest.runs[`${variant}_seed${seed}`].summary.successes,
      );
    }
  }
});

test("the difficult RL failure stays accessible and is not declared complete", () => {
  const entry = manifest.runs.posttrained_seed2_finish.trials[3];
  assert.equal(entry.success, false);
  const trace = json(entry.url);
  assert.ok(distanceMM(trace.state.at(-1), trace.goal) > 100);
});

test("manifest evidence links resolve and checkpoint metadata retain integrity", () => {
  for (const group of Object.values(manifest.evidence))
    for (const entry of group) {
      assert.ok(existsSync(resolve(publicRoot, entry.url.slice(1))), entry.url);
      if (entry.sha256) {
        assert.match(entry.sha256, /^[0-9a-f]{64}$/);
        assert.equal(
          createHash("sha256")
            .update(readFileSync(resolve(publicRoot, entry.url.slice(1))))
            .digest("hex"),
          entry.sha256,
        );
      }
    }
  assert.equal(manifest.clips.filter((c) => c.role === "train").length, 3);
  assert.equal(manifest.clips.filter((c) => c.role === "test").length, 1);
});
