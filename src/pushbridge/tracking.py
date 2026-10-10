import json
from pathlib import Path

import cv2
import numpy as np

from .data import sha256, transitions, write_json
from .video import VideoWriter


def marker(frame, color, min_area=20):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == 'red':
        # Narrow hue bands separate red paint from the orange/brown hand.
        # Phone compression shifts this painted cap toward magenta, while
        # shadows on the hand occupy the orange side of the red hue band.
        mask = cv2.inRange(hsv, (0, 85, 45), (3, 255, 255)) | cv2.inRange(hsv, (155, 85, 45), (179, 255, 255))
        # Join small gaps in hand-coloured paint without filling the whole image.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    elif color == 'blue':
        mask = cv2.inRange(hsv, (92, 85, 45), (132, 255, 255))
    else:
        raise ValueError(color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > frame.shape[0]*frame.shape[1]*0.08:
            continue
        if color == 'red':
            perimeter = cv2.arcLength(contour, True)
            circularity = 4*np.pi*area/max(perimeter*perimeter, 1)
            if circularity < 0.15:
                continue
            score = area*circularity
        else:
            score = area
        candidates.append((score, contour))
    if not candidates:
        return None
    contour = max(candidates, key=lambda item: item[0])[1]
    m = cv2.moments(contour)
    return np.array([m['m10']/m['m00'], m['m01']/m['m00']], dtype=np.float32)


def extract_video(video, calibration, preview):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 10:
        cap.release()
        raise ValueError(f'Video must have at least 10 frames/second: {video}')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if list(size) != calibration['image_size']:
        cap.release()
        raise ValueError(f'{video.name}: resolution differs from calibration; record with one fixed camera configuration.')
    h = np.array(calibration['homography'])
    writer = VideoWriter(preview, 10, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Cannot write tracking preview')
    states, times, valid = [], [], []
    index, next_time = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index/fps
            index += 1
            if timestamp + 0.5/fps < next_time:
                continue
            points = [marker(frame, 'blue'), marker(frame, 'red')]
            found = all(p is not None for p in points)
            state = np.zeros(4)
            if found:
                state = cv2.perspectiveTransform(np.array(points, dtype=np.float32).reshape(1, 2, 2), h)[0].flatten()
                found = bool(np.all(state.reshape(2, 2) >= [-0.02, -0.02]) and np.all(state.reshape(2, 2) <= [0.317, 0.23]))
            for p, color in zip(points, [(255, 150, 30), (40, 40, 255)]):
                if p is not None:
                    cv2.circle(frame, tuple(p.astype(int)), 9, color, 2)
            cv2.rectangle(frame, (0, size[1]-48), (size[0], size[1]), (25, 25, 25), -1)
            cv2.putText(frame, f'{next_time:.1f}s | '+('TRACKED' if found else 'MISSING - EXCLUDED'), (20, size[1]-17), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 130) if found else (80, 80, 255), 2)
            writer.write(frame)
            states.append(state)
            times.append(next_time)
            valid.append(found)
            next_time += 0.1
    finally:
        cap.release()
        writer.release()
    return {'state': np.asarray(states), 'time': np.asarray(times), 'valid': np.asarray(valid)}


def ingest(videos, calibration_path, out, collector, source='personal'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError(f'{out} already contains a dataset. Choose a new output folder to preserve it.')
    calibration = json.loads(Path(calibration_path).read_text())
    paths = sorted(p for p in Path(videos).iterdir() if p.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.m4v'])
    if not paths:
        raise ValueError(f'No videos found in {videos}')
    episodes, hashes = [], set()
    for i, path in enumerate(paths):
        digest = sha256(path)
        if digest in hashes:
            raise ValueError(f'{path.name}: duplicate video; duplicates cannot be independent train/test clips.')
        hashes.add(digest)
        ep = extract_video(path, calibration, out/f'{i:03d}_tracking.mp4')
        if len(ep['state']) < 2:
            raise ValueError(f'{path.name}: too short')
        n = len(transitions(ep)[0])
        if n < 15:
            raise ValueError(f'{path.name}: only {n} usable transitions. Inspect its tracking preview; markers must stay visible and move slowly.')
        detected = ep['state'][ep['valid']]
        displacement = np.linalg.norm(detected[-1, 2:]-detected[0, 2:])
        if displacement < 0.025:
            raise ValueError(f'{path.name}: object moved only {1000*displacement:.1f} mm. Record a push of at least 25 mm.')
        filename = f'{i:03d}.npz'
        np.savez_compressed(out/filename, **ep)
        episodes.append({'id': path.stem, 'file': filename, 'raw_name': path.name,
                         'raw_sha256': digest, 'valid_fraction': float(ep['valid'].mean()),
                         'transitions': n, 'object_displacement_mm': float(displacement*1000)})
        print(f'{path.name}: {n} transitions, {ep["valid"].mean():.0%} detected, {displacement*1000:.1f} mm travel')
    write_json(out/'manifest.json', {'schema_version': 1, 'source': source, 'collector': collector,
               'provenance_note': 'Personally recorded, as declared by operator; software cannot verify authorship.' if source == 'personal' else 'SYNTHETIC: software validation only; does not satisfy the challenge.',
               'calibration': calibration, 'calibration_sha256': sha256(calibration_path), 'episodes': episodes})
    print(f'Dataset saved to {out}. Inspect every *_tracking.mp4 before training.')
