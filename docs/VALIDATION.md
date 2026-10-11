# Validation and experimental results

The latest release checks are recorded at the end. Earlier sections preserve the sequence of software checks and single-video demonstrations; their measurements are separate from the learned personal-data results.

Run date: 9 October 2026. No personally recorded videos were available. These results do **not** satisfy the challenge's personal-data constraint and do not establish human-to-robot transfer.

## Environment and commands

Linux, Python 3.12.3, NumPy 1.26.4, OpenCV 4.6.0, CPU PyTorch 2.9.0, MuJoCo 3.3.7, imageio-ffmpeg 0.6.0. The available RTX 4060 was not needed. The local interpreter is provided through `run-local.sh` in this workspace; use the README installation steps on a new computer.

```bash
python -m pytest -q
pushbridge smoke --out results/smoke --episodes 8 --epochs 100
```

Seven automated checks passed: perspective calibration, coloured-marker tracking, exclusion of missing detections/large jumps, whole-episode splitting, dynamics rotation equivariance, physical pushing with an idle-object control, and encoded-video extraction with missing frames.

The calibration browser page was also operated through all four clicks and saved successfully. All five report videos loaded in the browser with 960 px width, nonzero durations and no media errors. Tracking and robot frames were visually inspected. H.264 encoding replaced an initial MPEG-4 export that the browser could not play.

## Data and predictive model

Ten synthetic 8-second clips are generated from MuJoCo with red object and blue tool markers. Each clip produced 79 usable 0.1-second transitions; both markers were detected in all sampled frames. The seed-7 whole-clip split contains six training, two validation and two test clips. Synthetic generation and robot evaluation use the same simulator, so this is deliberately favourable to the model.

Three ensemble members, each with 1,283 parameters, were fitted for at most 100 epochs. Test clips contain 158 transitions, of which 73 move the object by more than 1 mm.

| Test measurement | Observed value |
| --- | --- |
| Mean one-step displacement error | 0.64 mm |
| Zero-motion prediction baseline | 1.92 mm |
| Fixed-radius analytic contact baseline | 1.12 mm |
| Mean error on moving transitions | 1.37 mm |
| Five-step open-loop error | 2.99 mm |
| Ten-step open-loop error | 7.68 mm |
| Empirical scaled-radius coverage | 88.6% |

Open-loop predictions use the recorded tool actions but feed predicted object positions back into the model. Increasing error over the horizon is a real limitation. The radius was scaled on validation clips; test coverage is descriptive, not a formal guarantee.

## Paired simulation check

Eight trials use the same starts, goals and friction values for each controller. A success requires the object to stay within 18 mm of the goal after holding the tool still for one second. State observations come directly from MuJoCo.

| Controller | Successes | Mean final distance |
| --- | --- | --- |
| Unadapted replay | 0/8 | 117.3 mm |
| Demonstration prior | 8/8 | 14.8 mm |
| Geometric baseline | 8/8 | 12.7 mm |
| Learned-model MPC | 8/8 | 14.2 mm |
| MPC without uncertainty penalty | 8/8 | 14.2 mm |

An 8/8 success count has a Wilson 95% interval of approximately 68%-100%. The geometric baseline's final distance is better in this check. The uncertainty ablation ties the full planner. There is **no demonstrated control advantage over the geometric baseline**. The purpose of this run is to verify that the data-to-model-to-robot pipeline executes correctly and retains failures.

Exact metrics, raw synthetic clips, previews, learned weights, per-step trajectories and videos are stored under `results/smoke/`. `simulation_metrics.json` and `prediction_metrics.json` are the numerical source of this note. Controller latency depends on the computer and is reported in those files.

## Work outstanding after the initial synthetic check

Collect and inspect the applicant's real videos, train a new checkpoint with `--source personal`, evaluate it on held-out clips and paired robot trials, analyse failures, and publish the resulting repository and evidence. Installation from a fresh isolated environment and operation on Windows/macOS have not been verified. The existing workspace runtime was used for this check.

## First personal clip check - 10 October 2026

`IMG_5669.mov` is a user-provided 12.17-second recording at approximately 30 fps, with decoded dimensions 1440 × 1920. The cap moves toward the drawn target, and both markers are visible through the active push. The initial red detector sometimes selected or rejected a large hand-colour region. Its hue band was narrowed, small paint gaps are closed, and plausible red contours are ranked by compactness and area. A regression check for a cap beside a brown hand and an oversized distractor was added; all eight checks passed.

With the corrected tracker, the object was detected in all 122 sampled frames. Both markers were detected in all 85 samples between 2.5 and 11 seconds. Detections before/after that interval require visual inspection because the finger enters and leaves the image. The preview was checked visually. These are pixel detections, not calibrated world positions or verified contact measurements.

Some paper corners are outside the image, and the sheet's framing changes during the clip. The attempted feature-based background registration did not produce enough reliable inliers; no numerical camera-motion estimate or stabilisation is claimed. A fixed phone and a fully visible sheet are needed for the current calibration pipeline. This clip was not used to train a model. The synthetic results above were generated before this tracker adjustment and have not been rerun with it.

The diagnostic preview, pixel-coordinate tracks and quality report are saved under `results/first_personal_clip/` in this workspace. Original video remains at the supplied local path. Nothing was uploaded or published.

## Second personal clip - working robot retargeting

`IMG_5670.mov` was supplied on 10 October 2026. It is a 14.57-second recording at approximately 30 fps with dimensions 1440 × 1920. The cap and black target circle were detected in all 146 sampled frames; the finger was detected through a continuous 120-frame segment from 2.60 to 14.50 seconds. The paper edges remain outside the view. The fitted target circle provides a relative position/scale reference, with its radius normalised to a selected 20 mm simulator radius. No physical dimensions in the video are asserted.

The recording produced 26 object waypoints. A hand-coded follower with aligned recorded-finger proposals completed the path in 68 MuJoCo control steps. The final object was 3.3 mm from the simulated goal centre and 5.5 mm from the retargeted recorded endpoint. Mean simulated-path distance to the normalised reference was 2.2 mm; the reverse reference-to-simulation 90th percentile distance was 4.8 mm. These distances use the chosen simulator scale. They are not physical measurement errors in the original recording.

The side-by-side comparison aligns path progress and uses different source/simulation clocks. Initialisation sets the simulator's starting pose; all subsequent object motion arises from contact. Ten automated checks passed, including the new circle-detector and translation/zoom normalisation checks. Middle and final comparison frames were visually verified.

Circle-based ingestion exported this recording as one whole-clip learning episode with 119 usable transitions and 71 moving-object transitions. The training split correctly refuses to treat it as several independent episodes. No policy or world model has yet been trained on personal data. At least four more independent circle-based clips are needed for the existing minimum five-clip learning split; ten clips are recommended.

Outputs are under `results/second_personal_clip/`; the processed learning episode is under `data/processed/personal_circle/`. A local copy of the source is preserved under `data/raw/personal_circle/`. The source SHA-256 is recorded in the manifests and metrics. Nothing has been uploaded or published.

## Five personal clips - learned dynamics and control

Run date: 10 October 2026. Four additional user-supplied clips were combined with `IMG_5670.mov`. Each source has a distinct SHA-256 hash. The smaller recordings decode to 360 × 480 pixels; the earlier one is 1440 × 1920. Whole clips remain separate learning episodes.

The initial circle detector missed faint/compressed ink and sometimes fitted a small broken arc. It now uses a low-saturation mask with a more generous brightness limit, omits artificial contour points along image edges, requires at least 240° outline support and limits ellipse residual. Phone compression also shifted the cap toward magenta, while hand shadows entered the earlier red hue band. Revised hue bands resolved the observed identity swaps. Two regression checks cover the faint cropped circle, short-arc rejection and compressed cap colour beside skin. **Twelve automated checks passed.** These detector changes were made during data inspection before training or test evaluation.

Only the longest uninterrupted segment with finger, object and acceptable circle detections is retained. Later circle occlusions and broken outline fits are excluded, without bridging them. Three tracked frames per retained segment were visually inspected, along with the resulting object paths. The exact counts are in `data/processed/personal_five/manifest.json`.

| Source | Duration | Retained segment | Usable transitions | Moving transitions | Split |
| --- | --- | --- | --- | --- | --- |
| IMG_5670.mov | 14.57 s | 2.60-13.70 s | 111 | 63 | Test |
| Video.mov | 9.03 s | 1.70-5.70 s | 40 | 29 | Train |
| Video_1.mov | 7.23 s | 1.40-6.80 s | 54 | 35 | Validation |
| Video_2.mov | 4.50 s | 1.00-3.60 s | 26 | 15 | Train |
| Video_3.mov | 7.67 s | 0.90-6.70 s | 58 | 53 | Train |

Total: 289 transitions, 195 moving. Training: 124 transitions, 97 moving. The seed-7 split holds out entire recordings: `Video_3`, `Video` and `Video_2` train; `Video_1` validates; `IMG_5670` tests. The initial single-video retargeting run and its archived tracking outputs used an earlier detector version. Its 120-frame count is historical; the current learning extraction retains 112 frames.

Coordinates subtract the observed target centre, reverse image y, align the initial object-to-target direction and map the target's radius to a chosen 20 mm simulator radius. These are **normalised simulator metres**, not measured physical metres. This removes approximate translation/zoom; perspective and camera rotation remain uncorrected. Selected source timestamps are stored alongside a nominal 10 Hz learning grid.

```bash
pushbridge ingest-circle --videos data/raw/personal_circle \
  --out data/processed/personal_five --collector "User-supplied home recordings"
pushbridge train --dataset data/processed/personal_five \
  --out results/personal_five/model --epochs 200 --seed 7
pushbridge evaluate --checkpoint results/personal_five/model/world_model.pt \
  --out results/personal_five/evaluation --episodes 20 --seed 2026
python scripts/summarize_experiment.py --dataset data/processed/personal_five \
  --run results/personal_five --raw data/raw/personal_circle
```

The existing extraction folder is preserved; choose a new folder for re-extraction. Three 1,283-parameter ensemble members were fitted for at most 200 epochs. Each member bootstraps whole training clips and selects its checkpoint with validation error. The demonstration prior and raw replay use only training clips. No controller is trained in simulation. Learned effective contact ranges are 20.83, 20.68 and 20.72 mm in the chosen simulator frame. Training took approximately 1.3 seconds on CPU.

| Held-out test measurement | Observed value |
| --- | --- |
| Mean one-step vector error | 0.895 mm |
| 90th percentile error | 1.955 mm |
| Zero-motion baseline | 1.645 mm |
| Fixed analytic-contact baseline | 1.215 mm |
| Moving-transition mean error | 1.267 mm |
| Five-step open-loop error / windows | 4.059 mm / 22 |
| Ten-step open-loop error / windows | 7.922 mm / 11 |
| Empirical prediction-radius coverage | 99.1% |
| Mean prediction radius | 2.782 mm |

The test set is one recording with 111 correlated transitions. Validation mean vector error is 1.529 mm over 54 transitions, with 88.9% empirical radius coverage. The radius multiplier uses the validation error/disagreement ratio; neither these coverage measurements nor transfer to simulation have a formal coverage guarantee. Open-loop windows obey the same valid-transition and jump exclusions as one-step predictions.

Twenty paired simulation trials vary starts, goals and floor friction (0.15, 0.25 or 0.40). The robot uses exact simulator object/tool states and a different rounded contact tip. Each controller gets the same trial conditions. Success requires an object centre within 18 mm of the goal after one second with zero requested tool displacement.

| Controller | Successes | Mean final distance | p95 decision latency |
| --- | --- | --- | --- |
| Raw action replay | 0/20 | 121.54 mm | 0.058 ms |
| Demonstration prior | 20/20 | 15.23 mm | 0.228 ms |
| Geometry | 20/20 | 13.80 mm | 0.081 ms |
| Learned-model MPC | 20/20 | 12.80 mm | 7.262 ms |
| MPC without uncertainty penalty | 20/20 | 13.29 mm | 7.112 ms |

A 20/20 count has a Wilson 95% interval of approximately 83.9%-100%; replay's 0/20 interval is 0%-16.1%. MPC and geometry tie on success. The paired mean final-distance difference (geometry minus MPC) is 1.002 mm. A descriptive 10,000-resample paired bootstrap with seed 2026 gives an interval of −0.481 to 2.474 mm. It includes zero, so this small suite does not establish a control advantage. The uncertainty ablation also ties on success; its small final-distance difference does not establish a useful uncertainty effect.

**What worked:** the personal data trains a predictive model, its priors and predictions affect simulated robot actions, and held-out one-step errors improve on both prediction baselines. Circle-relative coordinates make cropped-paper views usable. **What did not:** raw action replay fails, longer model rollouts drift, occlusions discard portions of the recordings, and learned planning does not establish an advantage over the strong geometric baseline. Five clips from one object/surface do not cover diverse tasks or embodiments. Different contact geometry, approximate camera compensation and simulator-state observations limit the result.

The full report, PNG/SVG figures, tracking samples, learned weights, original-video hashes, all per-trial trajectories, failure videos and metrics are under `results/personal_five/` and `data/processed/personal_five/`. All five controller videos and their final frames were checked. Fresh dependency installation on other operating systems remains unverified. Nothing has been uploaded or published.

## SAC, imitation, MPC and classical feedback

The user requested an RL comparison on 10 October 2026. The first controlled extension uses Stable Baselines3 2.9.0 SAC with Gymnasium 1.4.0. PPO and TD3 were not trained. SAC fits the two continuous action dimensions and provides a straightforward simulator-training baseline. This remains state-based robot learning; the challenge's world-model option allows this scope without a VLA.

Six SAC models use seeds 0, 1 and 2 for two variants: scratch and personal-demo actor initialization. Every model receives 40,000 agent interaction steps. Two 64-unit layers are used for actor/critics; Adam learning rate is 0.0003, replay capacity 100,000, random warmup 1,000 steps, batch size 128, gamma 0.99, tau 0.005, one gradient update per four interactions and automatic entropy starting at 0.1. The initial log standard deviation is −1 for both variants.

The actor consumes `(tool-object)/0.05` and `(goal-object)/0.20`; actions map from two bounded values to the same 8 mm norm-limited tip moves as MPC. The reward is the change in `−100*object_goal_distance −10*tool_behind_pose_distance`, minus 0.01 per step, plus 10 for stable success and minus 5 outside the workspace. The behind pose is 23 mm behind the object along the goal direction. This dense shaping is engineered task knowledge, shared across the SAC variants.

The initializer uses 123 moving-finger action labels from `Video`, `Video_2` and `Video_3`; near-zero finger motion and already-reached goals are excluded. Observation/action pairs rotate together over ±60° with sixteen variants per label. Five hundred supervised updates with batch size 128 and learning rate 0.001 initialize the actor mean. No validation/test recording labels are used. Critic targets and replay use only robot simulation; human dynamics are not inserted as robot transitions. Imitation-only checkpoints are saved before simulator learning and evaluated for every seed.

Validation curves use ten seed-9012 tasks every 5,000 interactions. Final 40,000-step weights are used regardless of earlier validation peaks. The initial imitation-only actors solve 8/10, 5/10 and 7/10 validation tasks; scratch actors solve none. Early SAC updates reduce the initialized policies' success considerably, then the policies recover. All three initialized models end with 10/10 validation successes. Scratch validation ends at 8/10, 4/10 and 10/10. This is evidence of transient forgetting, not a consistent sample-efficiency improvement.

Every final policy receives the same twenty seed-2026 starts/goals/friction conditions used for MPC/geometry. All observe simulator state, use the same action bound and have the same 120-step cap. Success is checked after a one-second zero-action hold.

| Controller | Successes for seeds 0 / 1 / 2 | Mean final distance across models | p95 inference |
| --- | --- | --- | --- |
| SAC from scratch | 13/20 · 9/20 · 20/20 | 17.58 mm | 0.31 · 0.31 · 0.43 ms |
| Personal initialization + SAC | 20/20 · 20/20 · 20/20 | 15.60 mm | 0.40 · 0.45 · 0.52 ms |
| Imitation only | 16/20 · 11/20 · 18/20 | 32.20 mm | 0.52 · 0.45 · 0.46 ms |
| Geometry | 20/20 | 13.80 mm | See paired metrics |
| Learned-model MPC | 20/20 | 12.80 mm | See paired metrics |

Mean success across model seeds is 70% for scratch SAC, 100% for initialized SAC and 75% for imitation only. The same twenty tasks are reused for the three model replicates; they are not sixty independent task trials. The data improves final SAC performance at this fixed budget in this experiment. MPC and classical geometric/proportional feedback still match success and finish closer, without simulator policy training. The imitation-only baseline shows why improvement over scratch alone would not isolate RL's contribution.

Each SAC training run took approximately 75-77 seconds on CPU including validation, about 455 seconds total. The fixed budget counts agent decisions, while stable-success holds add physics control ticks. Scratch runs execute 45,690 / 44,940 / 46,180 total training control ticks; initialized runs execute 46,240 / 46,580 / 47,050. Validation trajectories are extra evaluation and never enter replay. Per-run times, hyperparameters, total ticks, initialized weights, final weights, learning curves and all paired trajectories are preserved under `results/rl_comparison/`.

Fourteen automated checks passed, including reproducible Gym resets, terminal stable-success handling and an explicit test that validation/test action labels cannot enter imitation. Gymnasium/SB3 environment checking also passed. A non-editable wheel was built and installed in a separate virtual environment sharing existing system packages, and its packaged MuJoCo assets successfully ran a physics trial. This does not verify a fully isolated dependency installation or other operating systems. Reports, prediction figures, controller videos and final frames were inspected. Nothing has been published.

The installed wheel also passed all fourteen checks in that separate environment, with OpenCV 4.11.0 and Matplotlib 3.11.2. Two additional failure videos replay saved actions through physics and exactly reproduce saved states (maximum difference 0 mm): scratch SAC seed 1, trial 2 stalls 25.26 mm from the goal with zero requested action in its last twenty steps; imitation seed 1, trial 1 leaves the workspace and ends 66.81 mm away. These are representative first failures, not worst-case selection. Their metadata and verified replays are linked in the RL report.

## Extended submission: personal post-training, compact transfer, prediction, speed

Completed on 10 October 2026, using the same five personal recordings and whole-clip split. All original results remain preserved.

Personal actor post-training after scratch simulator SAC used 200 updates, batch 128 and learning rate 0.0001. Critic parameters were checked bitwise unchanged. Before/after success counts were 13→20, 9→5 and 20→19 out of the same twenty source tasks for seeds 0/1/2. There were no new simulator **training** decisions. Model selection used a fixed update budget, not validation/test performance.

Compact transfer used 240/170/130 mm links, three bounded joints, a 3 mm tip, a 50 g puck, and 0.25 rad/s joint-command slew limits. Validation seed was 7131; test seed was 2027. Direct frozen SAC transfer achieved 19/0/9 successes. Adding contact normalization achieved 20/20/20. Additional 20,000-step target SAC achieved 20/20/19. All counts are out of twenty paired tasks per model; model replicas are not sixty independent task samples. Source geometry achieved 0/20; scaled source geometry 1/20; adapted MPC 17/20. A follow-up compact-specific classical controller achieved 14/20 after inspection on validation cases. Its post-comparison development is documented in the protocol. Final-distance means and individual cases are retained, including direct-transfer seed 1's near-threshold stalls.

Failure replays for direct-transfer seed 1, trial 0, and target-post-trained seed 2, trial 3, reproduced recorded state arrays with **zero maximum difference**. The latter failure ended 111.6 mm from the goal. Simulator object positions are set only by reset; all later frames come from physical contacts, not drawn object paths or teleportation.

Held-out state prediction was extended to twenty and forty steps: 13.80 and 21.82 normalised mm, compared with no-motion errors of 28.92 and 48.51 mm. There are only five and two non-overlapping windows respectively. The saved overlay initially exposed a source/control timestamp mismatch during browser QA; the loader was corrected to preserve original frame times, the video was regenerated, and a regression check now protects this distinction. No fitted world-model weights or earlier prediction metrics changed. The overlay is recorded video with state predictions projected onto it, not neural video generation.

Frozen-model robot diagnostics use achieved tool displacements. One-step mean errors were 0.855 mm on 693 source transitions and 0.864 mm on 1,000 compact transitions. Empirical validation-radius coverage was 97.0% and 96.4%; this is not a formal guarantee or a generalization claim across tasks.

Actor timing was repeated after training/render generation finished. Final local CPU medians were 0.175 ms (SB3), 0.045 ms (direct Torch) and 0.023 ms (NumPy). Relative speedups were 3.92× and 7.63×. Each path had 100 warm-up calls and 3,000 timed full action requests with alternating blocks; Torch used two threads. The standalone reconstructed Torch actor exactly matched all actions and states over twenty physics rollouts. NumPy matched all success outcomes; maximum action-vector difference was about 2.6 nanometres, while maximum full tool/object state-vector difference later in contact was 5.7 mm. This explicitly prevents inferring physical trace equivalence from small action errors alone.

The test suite has **18 passing tests**, including the compact kinematic Jacobian against MuJoCo, command limits/slew enforcement, original-video timestamp preservation, and independent actor reconstruction/action agreement. Stable Baselines3 `check_env` also passed for `CompactGym`.

The new dashboard was inspected in the in-app browser at its default viewport and at 390 × 844. Recording, strategy and seed selectors, prediction-window selection and range controls were exercised. Visible result counts matched JSON. All seven displayed video sources loaded with dimensions/durations and no media errors; both scientific SVG plots loaded. The phone layout had no page-level horizontal overflow. Tables scroll within their own wrappers. The report uses no external fonts, scripts or API calls, and includes a clipboard-selection fallback for local-file use.

A non-editable wheel was rebuilt and installed into the separate package-check virtual environment. Its import resolved to `site-packages`; all 18 tests passed there as well. The installed compact model and standalone exported actor completed a physical compact-arm trial successfully (17.90 mm final distance). The package-check environment shares the machine's Torch/NumPy runtime and is not a fully isolated or cross-platform installation claim. Inverting all 112 held-out observed object states back to their original recorded pixel coordinates gave a maximum numerical residual of 7.2e-13 pixels, confirming the state/frame transformation; this is an algebraic check, not tracking accuracy.


## Whole-object precision follow-up

The original 18 mm benchmarks are preserved. A separate follow-up requires the object centre within 8 mm throughout ten stationary hold ticks. A known-geometry controller aligns, tapers the final push and retracts before checking success. One validation run preceded the frozen protocol and evaluation.

Twenty paired cases across three checkpoint seeds compare policies alone and explicit hybrid feedback. An additional 20 fresh cases use seed 1 checkpoints. All 520 saved traces are checked for final distances, counted successes, complete successful holds, finite values and commanded joint speed no greater than 0.25 rad/s. The dashboard exposes all 360 paired policy/hybrid traces; its data are checked against their saved states and joint angles.

The package test suite now has 20 passing tests, including physical precision completion and the circular containment boundary. The latest dashboard was checked at desktop and 390 px mobile width, including replay, strategy/seed/task selection, tolerance presets and the retained RL failure. Existing learned weights and original numeric outcomes are unchanged. CPU CI is configured but has not run on GitHub yet.

The feedback-only baseline solves 18/20. Some incomplete runs are geometrically inside but lack release/verification before the decision budget expires. Hybrid RL seed 2 retains one approach failure; the controller never reaches its handoff. These cases are included, not relabelled as successes.

Six precision videos were re-rendered with readable titles and final-state badges. Every state, action, joint angle and phase was checked for bit-identical reproduction against its saved trace; the checks are in `results/precision_finish/video_verification.json`.

Run `python scripts/audit_submission.py` for artifact consistency checks. This audit does not verify tracking accuracy, real-world scale or human data authorship.


## Three.js presentation, 10 October 2026

The React/Vite app in `web/` uses the recorded 3R joints and object positions. Its four Node tests verify physical coordinate conversion, all 360 exported trace endpoints and successful holds, the retained RL failure, and evidence asset/hash integrity. The same 20 Python tests pass. The browser checks cover perspective and contact detail, playback/reset/pause, model and task selection, native video playback, research tabs, and a 390 px phone layout with no horizontal overflow. The presentation does not run a new physics engine or policy in the browser.

Five publication recording copies were remuxed without audio or location/device metadata. Full decoded-frame SHA-256 values, frame counts, frame rates and dimensions match the original files. `publication_verification.json` records the container and frame hashes. Original files are retained outside the repository; simulator checkpoints and trajectories were not modified.

## Longer SAC comparison and evidence reader, 11 October 2026

Six fixed-budget continuations add 40,000 SAC decisions to each of the three original target checkpoints under two terminal objectives. The total is 240,000 additional decisions and 292,430 physical control ticks, excluding validation. Actor, critic, entropy and optimizer state transfer, while replay starts empty because the parent checkpoints omitted replay data. The final checkpoint is used for every branch; validation never selects weights.

The original 18 mm objective yields strict policy-only counts 18/18/17 on the paired cases and 19/19/19 on newly declared fresh seed 11939. The stricter 8 mm hold objective yields 14/19/17 and 19/18/18. Every count is out of twenty cases per seed. Both new branches with the unchanged finisher score 20/20 in every seed and both suites. The policy alone still misses some finishes; the stricter objective is not universally better.

`audit_rl_followup.py` verifies all 760 follow-up traces, including 696 strict successes, six final weight hashes, training budgets, paired resets, command limits and stationary holds. All 120 paired original-baseline traces are bit-identical to the previous evidence. Three new native videos of the previously visible seed 2 Task 04 are physically re-simulated and checked against every recorded array, with maximum state difference zero.

The website now exports 600 paired replays, nine native MuJoCo videos and fourteen model files. All 360 original browser replay files are byte-identical to the preceding commit. The 22 Python tests and ten Node tests pass. Frontend tests verify geometry, successful holds, asset hashes, document routing, retention of all original and new trial outcomes, and exact replay links. Production build passes with the pinned lockfile.

The in-app browser checks cover the document reader, relative article links, the filtered original failure, longer policy versus longer hybrid replays, paired/fresh comparison controls, and native video metadata. A 390 × 844 layout has no page-level horizontal overflow; wide tables and document navigation scroll inside their own wrappers. All results remain simulation evidence using exact state, not a hardware or VLA evaluation.
