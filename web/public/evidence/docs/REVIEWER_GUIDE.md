# A two-minute tour of PushBridge

Can five home recordings of finger pushes help a robot push an object with a different arm and tip? PushBridge tests that question in MuJoCo, then separates what the learned policy accomplishes from what an explicit controller finishes.

The task is to place the red object entirely inside the green circle and keep it there for one second. Start with the Three.js app. To run it locally, follow [the web app guide](WEB_APP.md); the earlier [offline report](../index.html) remains available as a separate evidence view.

## 1. Watch one complete push

Click **Replay**. The default run uses the contact adapter with **+ Precision finish**. The robot approaches the object, pushes it, releases contact and checks that it stays inside. Use the scrubber to inspect a moment, or choose **Contact detail** for a close view.

The green target has a 20 mm radius and the red object a 12 mm radius. The object's centre must therefore finish within 8 mm of the target centre. A positive **Space to the boundary** means the whole object fits. A completed trial also needs ten zero-action control steps, or one second, inside that boundary.

These are saved physics results. The browser reconstructs the recorded joints and positions, with interpolation between samples. It does not run a new policy or invent a successful animation.

## 2. Compare the same task with and without help

Choose **Transfer unchanged** in the guided comparison, or select **Direct transfer** and **Policy only** yourself. This strategy sends the source policy's moves to the compact arm without correcting for its narrower tip. In this strict paired suite it completes 0/20 cases for each training seed. Switch to **+ Precision finish** to see the hand-coded controller take over near the goal and complete the same case.

Next choose **Adapt the contact**, which loads **Contact adapter** in **Policy only** mode. The adapter adjusts the tool-object coordinates for the changed tip radius while keeping the learned weights fixed. It completes 10/20, 20/20 and 20/20 cases for seeds 0, 1 and 2. Seed 1, loaded by this comparison, already completes the strict task without the finisher.

Finally choose **Find the RL failure**. **Adapter + RL** adds 20,000 training steps on the compact arm, but does not improve every result: even with the precision controller, seed 2 completes 19/20 cases. The button loads its remaining failure, task 04, which ends 111.6 mm from the goal. **Inspect a failed transfer** loads the same recorded case.

The selected-result explanation gives the endpoint and the corresponding outcome when feedback is added. Use the **Training seed** and **Paired task** selectors to inspect the full comparison. Each training seed faces the same twenty cases, so these are twenty repeated test conditions, not sixty independent tasks. The separate fresh-case result uses twenty new cases from the same task distribution and one training seed.

## 3. Trace the policy back to the recordings

Scroll to the five recordings and open a clip. The red and blue tracking overlays show the cap and fingertip positions used to create the motion samples. Three complete recordings are used for training, one for validation and one for testing. Adjacent frames never cross a split.

The four experiment tabs answer different questions:

- **Retargeting:** How can the same policy push with a shorter arm and narrower tip? The contact adapter changes relative coordinates; inverse kinematics turns the chosen tool move into joint commands.
- **Policy training:** Does a demonstration-based start help SAC learn? On the source arm, demonstration initialization followed by SAC reaches 20/20 cases for all three seeds. Fitting demonstrations after scratch SAC has mixed results: 13→20, 9→5 and 20→19.
- **Motion prediction:** Given the next finger moves, can a learned model predict the object's position? Choose a horizon to inspect error on one held-out recording. The overlay predicts positions on recorded footage; it does not generate video frames.
- **Inference speed:** Can the same actor run faster without changing its behavior? Direct Torch inference is 3.92× faster than the SB3 prediction wrapper on the measured CPU, with identical actions and states in all twenty compared rollouts.

The source-arm learning chart uses the original 18 mm centre-distance rule, which allowed partial overlap. The main replay and completion table use the stricter 8 mm full-containment rule plus the hold. Compare results within their stated protocol.

## 4. Check a result, then its limits

Open **Review the evidence** to find the experiment protocol, weights, all 520 saved trial outcomes and the audit. **Download this trace** gives the exact case selected in the main replay. **Original physics video** opens the MuJoCo reference video for task 01, seed 1; it does not change to match every selected case.

The personal recordings supply contact data and initial action examples. The robot policies themselves use exact simulator positions. This is a small state-based manipulation study, with a hand-coded precision finisher and no hardware, image-conditioned or language-conditioned policy evaluation. Classical feedback and MPC already solve the original source benchmark, and the less successful learning experiments remain included.

[README](../README.md) · [Detailed experiments](EXPERIMENTS.md) · [Precision protocol](PRECISION.md) · [Validation](VALIDATION.md)
