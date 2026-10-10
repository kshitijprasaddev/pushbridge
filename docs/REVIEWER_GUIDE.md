# A two-minute tour of PushBridge

The task is simple: move the red object completely inside the green circle, then keep it there for one second. The research question is whether a few personally recorded finger pushes can provide useful knowledge for a robot with different geometry.

## 1. Watch the finished task

Open [the report](../index.html) and choose **Demo**. The default replay uses a frozen contact-adapted policy followed by explicit precision feedback. Click **Replay task**, or move the slider. Positions and joint angles come from a saved MuJoCo run, not a browser animation invented to illustrate success.

Red is the object, green is the target, and blue is the robot's tip. Positive clearance means the object fits inside. The completion badge appears only at the end of a verified successful trace. The goal close-up preserves the same physical radius ratio as the main view.

## 2. Ask which component helps

Select **Direct transfer**, then **Learned policy alone**. The object stalls outside the circle. Switch on **Policy + precision feedback** to see an explicit controller finish the same task. The distance plot compares the two actual trajectories and labels the handoff.

Then select **Contact adapter**. That unchanged seed 1 policy already completes the strict task when allowed to continue. The original evaluator stopped at a looser boundary. The finisher is therefore a precision intervention, and its contribution is reported separately from policy learning.

The inspector covers all 20 paired tasks for every strategy, with and without feedback, across all three model seeds. It also reports a feedback-only baseline and 20 fresh task cases. Click **Inspect the remaining RL failure** to load that actual failed trajectory. The RL seed 2 hybrid still has a failure. Successes are counts on a small paired suite, not claims of universal reliability.

## 3. Audit the target rule

Open **Why did the old videos stop outside the circle?** Set the slider to 18 mm, then 8 mm. The geometry shows why the original rule permitted partial overlap. The counts re-score original saved endpoints only; they do not simulate new trajectories.

## 4. Follow the demonstration into the system

Under **Data**, inspect each original clip and its whole-recording split. **Transfer** explains how tool radii account for different contact clearances. **Learning** separates actor imitation, SAC interaction and personal post-training. **Prediction** overlays predicted states on held-out footage. **Speed** compares the same actor with and without inference wrappers.

## What the project demonstrates

It connects personal data, learned contact dynamics, reinforcement learning, geometry-aware transfer, terminal control and deployment checks in one reproducible study. The central result is an inspectable chain of evidence, including changes that did not improve performance.

The robot controller consumes exact simulator positions. It does not see images or receive language instructions. The final feedback controller is hand-coded. These are deliberate scope choices and are stated beside the results.

[Start with the README](../README.md) · [Detailed experiments](EXPERIMENTS.md) · [Precision protocol](PRECISION.md) · [Validation](VALIDATION.md)
