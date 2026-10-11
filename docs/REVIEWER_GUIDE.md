# A two-minute tour of PushBridge

**[Open the live research app → pushbridge.vercel.app](https://pushbridge.vercel.app/)**. No installation or sign-in required. Keep the app open while following this tour.

Can five home recordings of finger pushes help a robot push an object with a different arm and tip? PushBridge tests that question in MuJoCo, then separates what the learned policy accomplishes from what an explicit controller finishes.

The task is to place the red object entirely inside the green circle and keep it there for one second. Start with the 3D replay on the website. To run it locally, follow [the web app guide](WEB_APP.md); the earlier [offline report](../index.html) remains in the repository as a separate evidence view.

## 1. Watch one complete push

Click **Replay**. The default run uses the contact adapter with **+ Precision finish**. The robot approaches the object, pushes it, releases contact and checks that it stays inside. Use the scrubber to inspect a moment, or choose **Contact detail** for a close view.

The green target has a 20 mm radius and the red object a 12 mm radius. The object's centre must therefore finish within 8 mm of the target centre. A positive **Space to the boundary** means the whole object fits. A completed trial also needs ten zero-action control steps, or one second, inside that boundary.

These are saved physics results. The browser reconstructs the recorded joints and positions, with interpolation between samples. It does not run a new policy or invent a successful animation.

Use **MuJoCo footage** in the navigation to see the simulator's own rendering. Six reference videos compare the original strategies with and without the precision controller on task 01, training seed 1. Three more compare the earlier failed task 04, seed 2, before and after longer training and with the finisher. The gallery labels each video's model and case; it does not imply that a video exists for every 3D replay selection.

## 2. Compare the same task with and without help

Choose **Transfer unchanged** in the guided comparison, or select **Direct transfer** and **Policy only** yourself. This strategy sends the source policy's moves to the compact arm without correcting for its narrower tip. In this strict paired suite it completes 0/20 cases for each training seed. Switch to **+ Precision finish** to see the hand-coded controller take over near the goal and complete the same case.

Next choose **Adapt the contact**, which loads **Contact adapter** in **Policy only** mode. The adapter adjusts the tool-object coordinates for the changed tip radius while keeping the learned weights fixed. It completes 10/20, 20/20 and 20/20 cases for seeds 0, 1 and 2. Seed 1, loaded by this comparison, already completes the strict task without the finisher.

Finally choose **Find the RL failure**. **Adapter + RL** adds 20,000 training steps on the compact arm, but does not improve every result: even with the precision controller, seed 2 completes 19/20 cases. The button loads its remaining failure, task 04, which ends 111.6 mm from the goal. **Inspect a failed transfer** loads the same recorded case.

The selected-result explanation gives the endpoint and the corresponding outcome when feedback is added. Use the **Training seed** and **Paired task** selectors to inspect the full comparison. A training seed selects an independently trained set of weights, not a different goal. The three original runs have the same training budget but different randomness during learning. The paired task selects one of the same twenty start, goal and friction conditions used for all strategies and seeds.

The original 20,000-step target-training comparison is:

| Training seed | Contact adapter, policy only | Adapter + RL, policy only | Adapter + RL, with precision finish |
| --- | --- | --- | --- |
| 0 | 10/20 | 6/20 | 20/20 |
| 1 | 20/20 | 19/20 | 20/20 |
| 2 | 20/20 | 16/20 | 19/20 |

This distinction explains why changing a seed can change a replay so much. A successful example does not establish that the learned policy is reliable over the full suite, and the precision controller's success is credited separately. These are twenty repeated test conditions, not sixty independent tasks. The separate fresh-case result uses twenty new cases from the same task distribution and one training seed.

### What happened after longer training?

Under **Adapter + RL**, use the checkpoint selector to compare the original 20,000 target-training decisions with two 60,000-decision continuations. Both add 40,000 decisions to each original checkpoint. One keeps the original 18 mm training objective; the other uses the strict 8 mm containment rule during training. All three are evaluated with the same strict completion check.

| Target checkpoint | Paired policy-only successes, seeds 0 / 1 / 2 | New-case policy-only successes, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Original 20k | 6/20 · 19/20 · 16/20 | 9/20 · 19/20 · 16/20 |
| Longer 60k, original objective | 18/20 · 18/20 · 17/20 | 19/20 · 19/20 · 19/20 |
| Strict 60k, containment objective | 14/20 · 19/20 · 17/20 | 19/20 · 18/20 · 18/20 |

Longer training helps the weakest original seed substantially, but does not improve every checkpoint. Changing the terminal objective is not consistently better than keeping it. Both 60k branches complete all twenty cases for every seed in both suites **with the existing hand-coded finisher**. That result belongs to the hybrid policy and controller, rather than learned completion alone.

Keep seed 2 and task 04 selected to examine the earlier 111.6 mm failure. The longer, original-objective policy reaches 11.2 mm without the finisher, still outside full containment. With the finisher it reaches 2.8 mm and completes the release and hold. The strict-objective continuation also needs help on this case: its policy-only endpoint is 19.3 mm, versus 2.5 mm with the finisher.

The new-case suite uses seed 11939, declared before this follow-up, and tests all three checkpoints. It is separate from the original seed-9092 fresh check. These remain twenty cases from the same simple task distribution. Read [the follow-up methodology](../results/rl_precision_followup/README.md), [all outcomes](../results/rl_precision_followup/evaluation.json) and [the gained and lost cases](../results/rl_precision_followup/summary.json). The six continuations start with empty replay buffers because the original checkpoints did not include replay data; they are not uninterrupted 60,000-step runs.

## 3. Trace the policy back to the recordings

Scroll to the five recordings and open a clip. The red and blue tracking overlays show the cap and fingertip positions used to create the motion samples. Three complete recordings are used for training, one for validation and one for testing. Adjacent frames never cross a split.

The four experiment tabs answer different questions:

- **Retargeting:** How can the same policy push with a shorter arm and narrower tip? The contact adapter changes relative coordinates; inverse kinematics turns the chosen tool move into joint commands.
- **Policy training:** Does a demonstration-based start help SAC learn? On the source arm, demonstration initialization followed by SAC reaches 20/20 cases for all three seeds. Fitting demonstrations after scratch SAC has mixed results: 13→20, 9→5 and 20→19.
- **Motion prediction:** Given the next finger moves, can a learned model predict the object's position? Choose a horizon to inspect error on one held-out recording. The overlay predicts positions on recorded footage; it does not generate video frames.
- **Inference speed:** Can the same actor run faster without changing its behavior? Direct Torch inference is 3.92× faster than the SB3 prediction wrapper on the measured CPU, with identical actions and states in all twenty compared rollouts.

The source-arm learning chart uses the original 18 mm centre-distance rule, which allowed partial overlap. The main replay and completion table use the stricter 8 mm full-containment rule plus the hold. Compare results within their stated protocol.

## 4. Check a result, then its limits

Open **Review the evidence** to read the experiment protocol and inspect the results in the website's reader. Documents retain headings, tables, code and links; result JSON has a structured view and a raw-data option. The original files and weights remain available to download. **Download this trace** gives the exact case selected in the main replay. The original strict evaluation includes all 520 saved trial outcomes, including the failed approach above. The longer-training follow-up includes another 760 evaluated rollouts, with 120 paired original-baseline traces verified to match the old evidence exactly. Its audit checks all checkpoints, decisions, geometry and successful holds.

In the **MuJoCo footage** gallery, choose a recording to compare the strategies or the longer-training versions. Each recording labels its seed and task. MuJoCo produced the contacts and motion; Three.js presents the saved states.

The personal recordings supply contact data and initial action examples. The robot policies themselves use exact simulator positions. This is a small state-based manipulation study, with a hand-coded precision finisher and no hardware, image-conditioned or language-conditioned policy evaluation. Classical feedback and MPC already solve the original source benchmark, and the less successful learning experiments remain included.

[README](../README.md) · [Detailed experiments](EXPERIMENTS.md) · [Precision protocol](PRECISION.md) · [Validation](VALIDATION.md)
