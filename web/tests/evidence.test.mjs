import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  evidenceViewerUrl,
  resolveEvidenceLink,
  completionRows,
  headingId,
  replaySelection,
  trialReplayUrl,
  trialMethodLabel,
} from "../src/evidenceRoutes.mjs";

const manifest = JSON.parse(readFileSync("public/data/manifest.json", "utf8"));

test("articles and JSON open in the reader while model downloads stay unchanged", () => {
  const path = "/evidence/docs/REVIEWER_GUIDE.md";
  const url = new URL(
    evidenceViewerUrl(`${path}#define-a-complete-push`),
    "https://pushbridge.vercel.app",
  );
  assert.equal(url.searchParams.get("evidence"), path);
  assert.equal(url.hash, "#define-a-complete-push");
  assert.match(
    evidenceViewerUrl("/evidence/results/precision_evaluation.json"),
    /^\/\?evidence=/,
  );
  assert.equal(
    evidenceViewerUrl("/evidence/policies/source_sac_seed0.zip"),
    "/evidence/policies/source_sac_seed0.zip",
  );
});

test("repository-relative document links resolve to copied articles or GitHub source", () => {
  const link = resolveEvidenceLink(
    "PRECISION.md#what-counts",
    "docs/EXPERIMENTS.md",
    manifest,
  );
  assert.equal(
    new URL(link, "https://pushbridge.vercel.app").searchParams.get("evidence"),
    "/evidence/docs/PRECISION.md",
  );
  assert.equal(
    new URL(link, "https://pushbridge.vercel.app").hash,
    "#what-counts",
  );
  const trialLink = resolveEvidenceLink(
    "../results/precision_finish/evaluation.json",
    "docs/REVIEWER_GUIDE.md",
    manifest,
  );
  assert.equal(
    new URL(trialLink, "https://pushbridge.vercel.app").searchParams.get(
      "evidence",
    ),
    "/evidence/results/precision_evaluation.json",
  );
  assert.equal(
    resolveEvidenceLink(
      "../src/pushbridge/precision.py",
      "docs/EXPERIMENTS.md",
      manifest,
    ),
    "https://github.com/kshitijprasaddev/pushbridge/blob/main/src/pushbridge/precision.py",
  );
  assert.equal(
    resolveEvidenceLink("javascript:alert(1)", "docs/EXPERIMENTS.md", manifest),
    undefined,
  );
  assert.equal(
    resolveEvidenceLink(
      "//untrusted.example/file",
      "docs/EXPERIMENTS.md",
      manifest,
    ),
    undefined,
  );
});

test("trial table retains every original outcome without changing evidence", () => {
  const data = JSON.parse(
    readFileSync("public/evidence/results/precision_evaluation.json", "utf8"),
  );
  const before = JSON.stringify(data);
  const rows = completionRows(data);
  assert.equal(rows.length, 520);
  assert.equal(rows.filter((row) => row.success).length, 406);
  const missed = rows.find(
    (row) =>
      row.suite === "paired" &&
      row.method === "posttrained_seed2_finish" &&
      row.trial === 3,
  );
  assert.equal(missed.success, false);
  assert.ok(missed.final_distance_mm > 100);
  assert.equal(JSON.stringify(data), before);
  assert.equal(
    headingId("What is actually learned?"),
    "what-is-actually-learned",
  );
});

test("the follow-up explorer preserves all 760 trials and fresh-suite identity", () => {
  if (!manifest.rlFollowup) return;
  const data = JSON.parse(
    readFileSync(`public${manifest.rlFollowup.evaluationUrl}`, "utf8"),
  );
  const rows = completionRows(data);
  assert.equal(rows.length, 760);
  assert.equal(rows.filter((row) => row.success).length, 696);
  assert.ok(
    rows
      .filter((row) => row.suite === "fresh")
      .every((row) => row.suiteSeed === 11939),
  );
});

test("paired trial links load the exact model, task and controller while fresh cases stay separate", () => {
  const url = trialReplayUrl(
    { suite: "paired", method: "original_20k_seed2_finish", trial: 3 },
    manifest,
  );
  assert.deepEqual(
    replaySelection(new URL(url, "https://pushbridge.vercel.app").search),
    {
      strategy: "postRL",
      trainingVersion: "original",
      seed: 2,
      trial: 3,
      mode: "finish",
    },
  );
  assert.deepEqual(replaySelection("?run=strict_objective_seed0&task=20"), {
    strategy: "postRL",
    trainingVersion: "strict",
    seed: 0,
    trial: 19,
    mode: "policy",
  });
  assert.equal(
    trialReplayUrl(
      { suite: "fresh", method: "posttrained_seed2", trial: 3 },
      manifest,
    ),
    null,
  );
  assert.equal(
    trialReplayUrl(
      { suite: "paired", method: "feedback_only", trial: 3 },
      manifest,
    ),
    null,
  );
  assert.equal(replaySelection("?run=longer_original_seed5&task=4"), null);
  assert.equal(replaySelection("?run=posttrained_seed2&task=21"), null);
  assert.equal(
    trialMethodLabel("longer_original_seed2_finish"),
    "Longer RL 60k (18 mm goal) · seed 2 · with feedback",
  );
});
