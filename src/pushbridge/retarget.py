"""A personal-video demonstration without requiring visible paper corners.

The drawn target circle defines a relative visual task frame. Its observed
radius maps to a chosen 20 mm simulator radius: this is embodiment scaling,
not a measurement of physical distances in the recording. A hand-coded
waypoint follower drives the arm; no policy or world model is fitted here.
"""
from pathlib import Path
import html

import cv2
import numpy as np

from .data import sha256, transitions, write_json
from .policy import geometric_action
from .sim import PushEnv
from .tracking import marker
from .video import VideoWriter


def target_circle(frame):
    """Fit the black drawn circle in the lower-left part of this task's view."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Compression and exposure can lift thin black ink above dark-gray values.
    # Low saturation keeps the more generous brightness threshold off skin,
    # the coloured cap and the finger marker.
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = cv2.inRange(gray, 0, 135) & cv2.inRange(saturation, 0, 85)
    height, width = gray.shape
    mask[:int(height*.55)] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        points = contour[:, 0, :]
        # A clipped outline is closed along the image edge by findContours.
        # Those artificial edge points must not influence the ellipse fit.
        points = points[(points[:, 0] > 1) & (points[:, 0] < width-2) &
                        (points[:, 1] > 1) & (points[:, 1] < height-2)]
        if len(points) < 50:
            continue
        (x, y), (w, h), angle = cv2.fitEllipse(points)
        if not (width*.15 < min(w, h) and max(w, h) < width*.55 and min(w, h)/max(w, h) > .65):
            continue
        if y < height*.60 or x > width*.60:
            continue
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        local = ((points-[x, y]) @ rotation.T)/[w/2, h/2]
        radial = np.linalg.norm(local, axis=1)
        residual = np.median(np.abs(radial-1))
        angles = np.sort(np.arctan2(local[:, 1], local[:, 0]))
        coverage = 2*np.pi - np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
        # A small broken arc can fit an unrelated ellipse with a low residual.
        # Require at least two thirds of the outline, even for cropped circles.
        if residual > .04 or coverage < np.deg2rad(240):
            continue
        candidates.append((w*h, np.array([x, y]), np.sqrt(w*h)/2, float(residual)))
    if not candidates:
        return None
    _, center, radius, residual = max(candidates, key=lambda c: c[0])
    return center, radius, residual


def task_coordinates(points, centers, radii, frame, goal=(.225, .105)):
    relative = np.asarray(points)-np.asarray(centers)
    relative[..., 1] *= -1  # image down becomes tabletop up
    return np.asarray(goal) + (relative @ frame)*(.020/np.asarray(radii))[..., None]


def extract_task(video, out):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 10:
        raise ValueError('Video must have at least 10 fps')
    rows, index, next_time = [], 0, 0.
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (720, round(height*720/width))
    writer = VideoWriter(out/'tracking_preview.mp4', 10, size)
    detected_red = detected_blue = detected_goal = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            time_s = index/fps
            index += 1
            if time_s + .5/fps < next_time:
                continue
            red, blue, circle = marker(image, 'red'), marker(image, 'blue'), target_circle(image)
            detected_red += red is not None
            detected_blue += blue is not None
            detected_goal += circle is not None
            if red is not None and blue is not None and circle is not None:
                center, radius, residual = circle
                rows.append([time_s, *blue, *red, *center, radius, residual])
            display = cv2.resize(image, size)
            for point, label, color in [(red, 'object', (40, 240, 70)), (blue, 'finger', (255, 190, 20))]:
                if point is not None:
                    xy = tuple(np.round(point*720/width).astype(int))
                    cv2.circle(display, xy, 15, color, 3)
                    cv2.putText(display, label, (max(5, xy[0]-30), max(95, xy[1]-22)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            if circle is not None:
                xy = tuple(np.round(circle[0]*720/width).astype(int))
                cv2.drawMarker(display, xy, (50, 220, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.rectangle(display, (0, 0), (720, 76), (24, 29, 39), -1)
            cv2.putText(display, f'PERSONAL VIDEO | {time_s:.1f}s', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(display, 'Circle-relative tracking; physical dimensions unmeasured', (15, 59), cv2.FONT_HERSHEY_SIMPLEX, .45, (90, 200, 245), 1)
            writer.write(display)
            next_time += .1
    finally:
        cap.release()
        writer.release()
    if len(rows) < 20:
        raise ValueError('Too few frames with object, finger and drawn target circle visible. Inspect tracking_preview.mp4.')
    rows = np.array(rows)
    # Select the longest uninterrupted run. Missing detections are never bridged.
    breaks = np.flatnonzero(np.diff(rows[:, 0]) > .15)+1
    rows = max(np.split(rows, breaks), key=len)
    if len(rows) < 20:
        raise ValueError('No continuous tracked segment of at least two seconds')
    blue, red, centers, radii = rows[:, 1:3], rows[:, 3:5], rows[:, 5:7], rows[:, 7]
    relative = red[0]-centers[0]
    relative[1] *= -1
    direction = -relative/np.linalg.norm(relative)
    frame = np.stack([direction, [-direction[1], direction[0]]], axis=1)
    objects = task_coordinates(red, centers, radii, frame)
    fingers = task_coordinates(blue, centers, radii, frame)
    if np.any(objects < [.015, .015]) or np.any(objects > [.282, .195]):
        raise ValueError('Normalised object trajectory does not fit the simulator workspace; inspect circle tracking.')
    np.savez_compressed(out/'extracted_task.npz', video_time_s=rows[:, 0], finger_px=blue,
                        object_px=red, target_px=centers, target_radius_px=radii,
                        retargeted_object_m=objects, retargeted_finger_m=fingers, task_frame=frame)
    summary = {'sampled_frames': round(next_time*10), 'object_detections': int(detected_red),
               'finger_detections': int(detected_blue), 'target_detections': int(detected_goal),
               'continuous_segment_frames': len(rows), 'continuous_segment_s': [float(rows[0, 0]), float(rows[-1, 0])],
               'normalisation': 'Observed drawn-target radius maps to a chosen 20 mm simulator radius. This does not measure real physical distances.',
               'target_radius_px_range': [float(radii.min()), float(radii.max())]}
    return rows[:, 0], objects, fingers, summary


def waypoint_indices(path, spacing=.006):
    indices = [0]
    for i in range(1, len(path)):
        if np.linalg.norm(path[i]-path[indices[-1]]) >= spacing:
            indices.append(i)
    if indices[-1] != len(path)-1:
        indices.append(len(path)-1)
    return np.array(indices)


def source_frame(cap, time_s, size=(480, 640)):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s*1000)
    ok, image = cap.read()
    if not ok:
        return np.zeros((size[1], size[0], 3), np.uint8)
    factor = min(size[0]/image.shape[1], size[1]/image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*factor), round(image.shape[0]*factor)))
    canvas = np.zeros((size[1], size[0], 3), np.uint8)
    x, y = (size[0]-resized.shape[1])//2, (size[1]-resized.shape[0])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


def retarget_demo(video, out, max_steps=600):
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    video, out = Path(video), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    times, object_path, finger_path, tracking = extract_task(video, out)
    selected = waypoint_indices(object_path)
    waypoints = object_path[selected]
    if len(waypoints) < 5:
        raise ValueError('Recorded object path contains too little motion')
    env = PushEnv()
    initial_side = finger_path[0]-object_path[0]
    initial_side /= max(np.linalg.norm(initial_side), 1e-8)
    tool = np.clip(object_path[0]+initial_side*.045, [.015, .015], [.282, .195])
    state = env.reset(object_path[0], tool, [.225, .105])
    writer = VideoWriter(out/'robot_demo.mp4', 10, (960, 640))
    paired = VideoWriter(out/'comparison.mp4', 10, (1440, 640))
    source = cv2.VideoCapture(str(video))
    states, actions, progress, video_times, used_finger_proposals = [state.copy()], [], [], [], []
    cursor = 1
    complete = False
    try:
        for step in range(max_steps):
            while cursor < len(waypoints)-1 and np.linalg.norm(state[2:]-waypoints[cursor]) < .008:
                cursor += 1
            target = waypoints[cursor]
            action = geometric_action(state, target)
            # Use the demonstrated finger side when it agrees with the direction
            # to the next recorded object waypoint. Otherwise the explicit
            # geometric controller repositions the robot around the object.
            side = finger_path[selected[cursor]]-object_path[selected[cursor]]
            side /= max(np.linalg.norm(side), 1e-8)
            error = target-state[2:]
            direction = error/max(np.linalg.norm(error), 1e-8)
            aligned = np.dot(-side, direction) > .80
            used = False
            if aligned and np.linalg.norm(state[:2]-(state[2:]+side*.021)) < .009:
                desired = state[2:]+side*.019 + direction*min(.004, np.linalg.norm(error))
                action = desired-state[:2]
                used = True
            state = env.step(action)
            actions.append(action.copy())
            states.append(state.copy())
            progress.append(cursor)
            video_times.append(times[selected[cursor]])
            used_finger_proposals.append(used)
            frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (960, 82), (24, 29, 39), -1)
            cv2.putText(frame, f'PERSONAL VIDEO RETARGETING | waypoint {cursor+1}/{len(waypoints)}', (18, 31), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(frame, 'Hand-coded follower; no policy training | actual MuJoCo contact', (18, 64), cv2.FONT_HERSHEY_SIMPLEX, .50, (100, 210, 245), 1)
            writer.write(frame)
            left = source_frame(source, times[selected[cursor]])
            cv2.rectangle(left, (0, 0), (480, 80), (24, 29, 39), -1)
            cv2.putText(left, 'YOUR RECORDING', (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(left, f'Source {times[selected[cursor]]:.1f}s | matched path progress', (12, 60), cv2.FONT_HERSHEY_SIMPLEX, .44, (100, 210, 245), 1)
            paired.write(np.hstack([left, frame]))
            if step == 10:
                cv2.imwrite(str(out/'demo_preview.jpg'), np.hstack([left, frame]))
            if cursor == len(waypoints)-1 and np.linalg.norm(state[2:]-target) < .008:
                complete = True
                break
    finally:
        writer.release()
        paired.release()
        source.release()
        env.close()
    trajectory = np.array(states)
    distances = np.linalg.norm(trajectory[:, None, 2:]-object_path[None], axis=-1)
    goal_distance = np.linalg.norm(state[2:]-[.225, .105])
    report = {'source': 'user-provided personal recording', 'video': video.name, 'raw_sha256': sha256(video),
              'tracking': tracking, 'controller': 'Recorded object waypoints with geometric feedback and aligned recorded finger-side proposals.',
              'policy_training': False, 'world_model_training': False,
              'simulation_observation': 'Ground-truth MuJoCo object/tool positions',
              'waypoints': len(waypoints), 'simulation_control_steps': len(actions),
              'recorded_finger_proposal_steps': int(sum(used_finger_proposals)),
              'recorded_path_completed': complete,
              'final_distance_to_retargeted_endpoint_mm': float(np.linalg.norm(state[2:]-object_path[-1])*1000),
              'final_distance_to_sim_goal_mm': float(goal_distance*1000),
              'within_sim_goal_radius_20mm': bool(goal_distance < .020),
              'mean_sim_to_reference_path_distance_mm': float(distances.min(axis=1).mean()*1000),
              'reference_path_to_sim_p90_distance_mm': float(np.quantile(distances.min(axis=0), .9)*1000),
              'limitations': ['Single clip; no success-rate or generalisation claim.', 'Circle fits compensate translation and approximate scale changes, not full camera perspective/rotation.',
                             '20 mm goal radius is chosen for embodiment normalisation, not measured in the recording.',
                             'Path errors are measured against the normalised reference, not real-world physical distances.',
                             'Comparison video aligns path progress; source and simulation timing are different.',
                             'The follower and contact rules are hand-coded. This is not a trained VLA or learned dynamics policy.']}
    np.savez_compressed(out/'robot_trajectory.npz', state=trajectory, requested_action=np.array(actions), waypoint=np.array(progress), matched_video_time_s=np.array(video_times))
    write_json(out/'retarget_metrics.json', report)
    make_retarget_report(report, out)
    print(f'Personal-video path completed: {complete}; final sim goal distance: {goal_distance*1000:.1f} mm; {len(actions)} steps', flush=True)
    return report


def make_retarget_report(report, out):
    out = Path(out)
    image = '<img src="trajectory_plot.png" alt="Recorded and simulated object paths" style="width:100%;max-width:800px">' if (out/'trajectory_plot.png').exists() else ''
    page = f'''<!doctype html><html><meta charset="utf-8"><title>PushBridge personal-video demonstration</title>
<style>body{{font:17px system-ui;background:#101a27;color:#eee;max-width:1150px;margin:36px auto;padding:0 24px}}.source{{background:#29354b;border-left:5px solid #79ceb1;padding:16px}}video{{width:100%;border-radius:8px;margin:15px 0}}a{{color:#9fdbda}}.metrics{{display:flex;gap:35px;flex-wrap:wrap;margin:24px 0}}.metrics b{{display:block;font-size:30px}}p{{line-height:1.5}}</style>
<h1>Personal video → simulated robot</h1><p class="source">Source: {html.escape(report['video'])}, supplied by the user. Recorded motion drives a physical MuJoCo pushing arm.</p>
<video controls muted preload="metadata" poster="demo_preview.jpg" src="comparison.mp4"></video>
<p>Left: the supplied recording. Right: the robot following its retargeted object path. The views align path progress; recording and simulation clocks differ.</p>
<div class="metrics"><div><b>{report['waypoints']}</b>recorded waypoints</div><div><b>{report['simulation_control_steps']}</b>robot control steps</div><div><b>{report['final_distance_to_sim_goal_mm']:.1f} mm</b>final simulated goal distance</div></div>
<p>The drawn black circle anchors position and scale. Its image radius maps to a chosen 20 mm simulator radius; real physical dimensions were not measured. A hand-coded controller follows the recorded waypoints and uses the demonstrated finger side for aligned contact proposals. This run did not train a policy or world model.</p>
{image}
<p>This is one demonstration, with ground-truth simulator observations. It establishes a working data-to-robot connection; it does not establish learned-policy performance, a success rate or generalisation.</p>
<p><a href="retarget_metrics.json">Full metrics and limitations</a> · <a href="tracking_preview.mp4">Marker and target-circle tracking</a> · <a href="robot_demo.mp4">Robot video</a></p></html>'''
    (out/'report.html').write_text(page)


def ingest_circle(videos, out, collector):
    """Whole-clip training episodes in the explicitly normalised task frame."""
    videos, out = Path(videos), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Dataset already exists; choose a new output folder to preserve it')
    paths = [videos] if videos.is_file() else sorted(p for p in videos.iterdir() if p.suffix.lower() in ['.mov', '.mp4', '.m4v', '.avi', '.mkv'])
    if not paths:
        raise ValueError('No videos found')
    items, digests = [], set()
    for i, path in enumerate(paths):
        digest = sha256(path)
        if digest in digests:
            raise ValueError(f'{path.name}: duplicate source video')
        digests.add(digest)
        folder = out/f'{i:03d}_diagnostics'
        folder.mkdir(exist_ok=True)
        source_times, objects, fingers, tracking = extract_task(path, folder)
        episode = {'state': np.c_[fingers, objects], 'time': np.arange(len(source_times))*.1,
                   'valid': np.ones(len(source_times), dtype=bool)}
        state, actions, delta, _ = transitions(episode)
        if len(state) < 15 or np.sum(np.linalg.norm(delta, axis=1) > .001) < 5:
            raise ValueError(f'{path.name}: too little usable motion; inspect its preview')
        filename = f'{i:03d}.npz'
        np.savez_compressed(out/filename, **episode, video_time_s=source_times)
        items.append({'id': path.stem, 'file': filename, 'raw_name': path.name, 'raw_sha256': digest,
                      'valid_fraction': 1.0, 'transitions': len(state), 'moving_transitions': int(np.sum(np.linalg.norm(delta, axis=1) > .001)),
                      'tracking': tracking, 'coordinate_units': 'normalised simulator metres, not measured real metres'})
        print(f'{path.name}: {len(state)} usable transitions, {items[-1]["moving_transitions"]} moving transitions', flush=True)
    write_json(out/'manifest.json', {'schema_version': 1, 'source': 'personal', 'collector': collector,
               'provenance_note': 'User/operator declares these recordings personal; software cannot verify authorship.',
               'calibration': {'method': 'black target-circle relative coordinates', 'sim_goal_radius_m': .020,
                               'metric_scale_measured': False, 'sample_hz': 10, 'time_note': 'Nominal 10 Hz dataset grid; original selected frame times retained in video_time_s.'},
               'episodes': items})
    print(f'Saved {len(items)} personal clip(s) to {out}. Training requires at least five independent clips.')
