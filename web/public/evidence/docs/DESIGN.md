# Design and experiment notes

## Why this scope

The hard constraint is applicant-collected data affecting a simulated manipulator. A home tabletop pushing experiment makes that dependency observable: replace the recordings, and both the learned dynamics and demonstration prior change. Cheap coloured markers make tracking failures inspectable. A compact state world model can be fitted and evaluated today without large-model fine-tuning.

The two-revolute-joint arm is a deliberately different embodiment from the human finger. Its articulated links and position servos drive a real contacting fingertip in MuJoCo. It is simpler than a Panda arm or LIBERO environment, but satisfies the manipulator-in-simulation part of the brief once personal data is used. Sim-to-real operation is outside the prototype's scope.

## Working circle-relative retargeting

The second personal recording provided a readable drawn target circle but no visible paper corners. The system now supports a task-relative frame: fit the black circle, subtract its centre from observed marker positions, map its radius to a selected simulator radius, and rotate the initial object-to-goal direction onto the arm's task axis. This compensates image translation and approximate uniform zoom, but cannot recover real physical scale or full projective camera motion.

The recorded object trajectory selects robot waypoints. A geometric controller follows them, with additional proposals from the recorded finger-contact side when it agrees with the desired movement. This completed one real-video-driven physical rollout. It is an explicitly hand-coded retargeting demonstration. Circle-based ingestion now also feeds the learned dynamics experiment: five supplied recordings produce 289 usable transitions and a three/one/one whole-clip split.

## What would make a useful result

Report all controllers, including the geometry baseline. A planner outperforming replay is expected because replay does not adapt its trajectory to a new goal. Evidence that the world model helps would require an advantage over geometry or the demonstration prior, or better predicted outcomes of held-out actions. Matching a simple baseline is still an honest result if the model's transfer limits are analysed.

Prediction quality and control quality are different measurements. The model may improve one-step errors while accumulating long-horizon drift. An uncertainty penalty may have no effect when all ensemble members agree. The validation radius only scales ensemble disagreement using measured residuals; it does not detect all domain shift.

## Personal-data experiment protocol

1. Use whole-clip splits, not random frames. The completed dataset has five clips; ten would make a more useful split. Preserve raw files, hashes, coordinate assumptions and tracking/setup images.
2. Inspect each tracking preview. Exclude bad clips before examining test metrics, record exclusions and re-record if the dataset becomes too small.
3. Fit with seed 7; preserve all training, validation and test IDs in the checkpoint. Five clips give three/one/one; ten clips give six/two/two.
4. Report one-step test error, moving-frame test error, five-/ten-step open-loop error and the two prediction baselines.
5. Evaluate all five controllers on the same twenty starts/goals/friction values with seed 2026. Keep failures and report uncertainty intervals.
6. Watch failures. Separate tracking errors, inaccurate learned contact dynamics, servo tracking errors, and controller proposal failures.
7. If time permits, collect a second surface/object dataset and evaluate a fixed model on it. Keep this dataset out of training and checkpoint selection. Treat this as an additional transfer experiment, rather than quietly fitting to it.

## Reinforcement learning comparison

The RL extension compares SAC from scratch with an actor initialized by imitation of the three training recordings. Relative finger/object/goal features and action labels rotate together during augmentation. Only actor means receive this supervised initialization. Critic targets and replay come exclusively from the robot simulator, avoiding a false assumption that human trajectories obey the robot's contact physics.

Both SAC variants have the same network, exploration initialization, simulator-interaction budget and dense geometric reward. Three training seeds expose variability. The final 40,000-step weights are used even when earlier validation checkpoints look better. An imitation-only baseline separates video learning from additional simulator learning. Geometry is classical proportional tip-position feedback backed by joint PD servos; MPC includes geometric proposals and receives no simulator training. Comparing their success counts without this resource distinction would be misleading. These algorithms consume states; a VLA would add image and language conditioning.

## Known modelling assumptions

- A4 dimensions are fixed; metric scale changes if another paper size is used.
- A single planar homography handles perspective but not fingernail height or camera movement.
- Marker detection uses fixed HSV thresholds. The red band includes the magenta shift observed in compressed clips and excludes orange hand shadows. Blue candidates are ranked by area; red candidates are ranked by area times circularity. Oversized regions are excluded individually. Previews are required to verify identity.
- The black circle uses low-saturation dark pixels, ellipse fits and outline-coverage checks. Cropped or occluded contours can still be rejected; only uninterrupted detections become learning segments. No full camera-pose correction is claimed.
- Video timing is approximated by the file's nominal frame rate, then sampled at 10 Hz. Variable-frame-rate capture can introduce timing error.
- State excludes velocity; the model assumes quasi-static pushes. Inertia after a stationary finger command is not represented.
- Fingernail positions are a proxy for fingertip contact. A learned contact radius is an effective parameter, not a physical measurement of finger size.
- The model is rotation-equivariant by construction; that does not establish transfer across friction or object shapes.
- MPC combines learned predictions and a demonstration prior with geometric action proposals and a hand-designed terminal cost. Ablations are necessary before claiming learned control benefits.
- Simulation uses exact object/tool positions, a fixed-size puck and a planar workspace. Image observations, 3D manipulation and obstacle avoidance are not implemented.

## Software verification already performed

The synthetic generator uses the same simulator later used for evaluation. This is intentionally an easy, biased software check. It is separate from the completed personal-data experiment. See `VALIDATION.md` for both sets of observed measurements. The personal-data model improves held-out one-step prediction errors, but its planner, the demonstration prior and geometry all achieve 20/20 on the simple paired task suite. A small mean final-distance improvement has a paired bootstrap interval including zero. No control advantage or broad generalisation is established. Twelve software checks and tracking/video inspection cover the current code; the README records reproducible commands.

## Personal post-training and embodiment experiments

The submission now tests two distinct placements of personal action supervision. The original experiment initializes an actor before 40,000 SAC interactions. The follow-up loads a scratch SAC policy after those interactions and applies 200 supervised actor updates at learning rate 0.0001. Both use the same training-only labels and rotation augmentation. The latter transfers simulator pretraining, not an external VLA. Its critics remain unchanged; no human tuples are presented as robot transitions. Fixed budgets prevent selecting the most favorable update count from test outcomes.

A second physical embodiment makes the task-frame choice concrete. `CompactPushEnv` has three planar hinge joints and links of 240/170/130 mm, compared with the source arm's 320/300 mm. A 3 mm tip replaces the 8 mm tip; the object mass doubles to 50 g. Joint targets obey known joint bounds and a 0.25 rad/s **command slew** limit. Actual joint velocities are governed by the PD servos and soft physical constraints; the command limit is not a hard measured-velocity guarantee.

The inverse-kinematics adapter solves a damped Cartesian Jacobian system and adds a null-space preference for a home posture. It clips iterative updates to joint bounds and interpolates bounded commands over fifty 2 ms MuJoCo steps. The implementation's analytic kinematics and Jacobian are tested against MuJoCo's independently computed site pose and Jacobian. Tracking residuals and slew-limited decisions are logged per task.

The task-space contact adapter maps tool/object relative displacement by 20/15, the ratio of source and target contact radii, then divides requested actions by the same factor. Object position and goal displacement remain in the common task frame. This is an engineered contact-clearance normalization, not an exact physical similarity transform. It does not compensate mass, friction, compliance or servo lag. Its utility is measured against direct frozen-policy transfer, not assumed.

Target SAC starts from the source's personal-initialized actor and trained critics/optimizers. Its target replay starts empty; target tuples come only from compact-arm physics. It receives 20,000 extra target decisions per seed. Reward shaping uses an 18 mm behind-object pose appropriate to the narrow tool. Validation uses seed 7131, while the twenty fixed target tests use seed 2027. Final fixed-budget checkpoints are retained, including the regression at seed 2. Source and target suites use different seeds, so controller comparisons are paired within each suite.

The original geometry baseline is retained. A follow-up compact-aware controller uses known contact dimensions, 4 mm pushes and tighter lateral correction. It was implemented after the initial transfer results and checked on validation cases; this chronology is disclosed. Baseline improvements and learned-policy failures remain visible.

## Inspectable prediction and deployment

Action-conditioned object states are predicted on one held-out personal clip at horizons of 1/5/10/20/40 steps. Finger motion is supplied from the recording throughout each rollout; predicted object state is not reset inside the window. Non-overlapping windows avoid claiming every overlapping segment is independent, but they still come from one correlated recording. Error counts decrease to just two windows at forty steps.

The state-to-image projection inverts the same task-frame rotation and per-frame circle scale used during extraction. It uses original `video_time_s`, separately from the nominal 10 Hz control grid. These overlays are visualizations of state predictions on recorded frames; there is no generative image/video network. Robot-domain checks use achieved, rather than requested, tool displacements because servo lag would otherwise confound the dynamics input. They do not update the learned model. Accumulated one-step error radii are only a heuristic, not calibrated open-loop coverage.

Deployment keeps the trained float32 weights. The direct Torch actor reconstructs two ReLU layers plus a tanh mean-action head from an 18 KB NPZ and applies the original SB3 action unscaling, observation encoding and movement clipping. It removes critic, distribution and training wrappers from prediction without changing tested Torch arithmetic. A NumPy path uses the same weights with a different arithmetic order. Tiny action-rounding changes can alter later physical contact or the step at which the success-hold threshold is crossed, so action equivalence alone is insufficient. Twenty full paired physical traces distinguish exact Torch reproduction from NumPy outcome agreement.

The static dashboard consumes generated measured JSON/JS, not fabricated figures or a server-side simulator. Selectors update real saved run videos and metrics; a range control inspects saved world-model rollouts. The same JSON is embedded as a local script so opening `index.html` from the ZIP also works without file-origin fetch access. Training and simulation remain Python-only; HTML/CSS/JavaScript are presentation code.
