# Record your dataset at home

**Task: push a red bottle cap into a target circle using your index finger.** You do not need a robot, headset, special camera or GPU.

## Circle-based capture used by the working personal demo

The second supplied clip, `IMG_5670.mov`, now drives the robot through circle-relative retargeting. To collect more data for this method, keep the phone fixed, put the full black target circle in the lower-left part of the image, and start the red cap above/right of it. Keep the blue marker and red cap visible through each push. Record separate attempts with varied starts, approach angles and speed, while keeping the circle and camera setup unchanged. Aim for ten clips; five is the minimum for whole-clip training/validation/test splits.

The circle-based method does not need visible paper edges or a measured paper size. It normalises the drawn circle to a chosen simulator size; it does not recover physical distances from the recording. Use `ingest-circle` as described in the README. The instructions below describe the alternative A4 method for metric planar calibration.

## Materials and setup

- One A4 sheet (297 × 210 mm), tape and a pen.
- A small light cap or block, ideally round and about 24 mm wide. It should slide, not roll.
- Red paper/tape on the object, blue paper/tape or washable ink on your index fingernail.
- A phone with enough space for ten short original video files.

Tape the A4 sheet flat, landscape. Mark the four corners in black. Draw a target circle about 4 cm wide near the right edge, leaving 4 cm clearance from the edge. Put the object near the left side. The optional [printable sheet](recording-sheet.svg) has corner guides and a target; print landscape at **100% / actual size**, without scaling. The calibration points are the physical paper corners, not the inset marks.

Fix the phone looking down at the whole sheet. A stable support or a securely taped mount works; keep it out of the pushing area. The lens should be approximately centred over the sheet and as close to vertical as practical. Use the rear camera, normal 1080p video at 30 fps, no slow motion and no zoom. Use bright diffuse light, and avoid other red/blue objects in the frame. Lock focus/exposure if your phone allows it. Keep the phone position and orientation fixed for **all** clips.

The blue mark should be close to the fingertip and as low as practical. Keep the finger nearly horizontal. Tracking the fingernail above the paper introduces parallax; this is a measured proxy for the contact point, not exact 3D hand motion. Our learned contact range absorbs some marker offset, but cannot undo a moving camera.

## Ten clips, 5-8 seconds each

1. Start recording with the object and blue marker visible. Pause for half a second.
2. Put the fingertip just behind the object; use the fingertip to slide it toward the circle.
3. Push slowly and continuously over 3-5 seconds. Do not pick the object up.
4. Keep your hand clear of the red top marker. Avoid a curled hand hiding it.
5. Let the object settle for half a second, then stop recording.
6. Reset between clips. Do not put a reset or a second attempt into the same clip.

For clips 1-4, start slightly above/below the sheet centre and push to the circle. For 5-8, vary the starting distance, angle and speed. For 9-10, include a slightly off-centre contact or imperfect push. Every clip should move the object at least 3 cm. The demonstration prior uses the **actual reached endpoint**; the system does not infer that a failed attempt reached the intended target.

Keep original files and filenames. Copy them into `data/raw/personal/` in the project or tell Codex their local folder path. Do not send clips through a messenger that compresses/resizes/rotates them. Take one setup photo and note your object size, table material, lighting, and recording date for the README. Keep faces and personal information outside the frame if you intend to publish the recordings.

## What happens next

Run `pushbridge calibrate` on one clip. Click the sheet's physical **top-left, top-right, bottom-right, bottom-left** corners in that order. Run `ingest`, then watch every tracking preview. A ring should follow the red object and the blue finger marker correctly. Missing frames are excluded; a wrong coloured object being tracked is not automatically detectable. Re-record or remove bad clips before training.

Training holds out whole clips. The learned model predicts object motion from finger position and displacement. A planner transfers that model to a physical MuJoCo pushing arm. Both the model errors and robot success rate are reported, including simpler baselines.
