import json

import cv2
import numpy as np
import pytest
import torch

from pushbridge.calibration import save_calibration
from pushbridge.data import episode_split, transitions
from pushbridge.model import ContactDynamics
from pushbridge.policy import geometric_action
from pushbridge.sim import PushEnv
from pushbridge.tracking import extract_video, marker
from pushbridge.retarget import target_circle, task_coordinates


def test_perspective_calibration_maps_corners_to_metres(tmp_path):
    frame = np.zeros((540, 720, 3), np.uint8)
    corners = [[80, 50], [660, 90], [620, 490], [70, 450]]
    path = tmp_path/'calibration.json'
    save_calibration(frame, corners, path)
    h = np.array(json.loads(path.read_text())['homography'])
    projected = cv2.perspectiveTransform(np.array([corners], dtype=np.float32), h)[0]
    np.testing.assert_allclose(projected, [[0, .21], [.297, .21], [.297, 0], [0, 0]], atol=1e-6)


def test_detection_tracks_markers_and_missing_marker_is_none():
    image = np.zeros((180, 240, 3), np.uint8)
    cv2.circle(image, (55, 90), 10, (230, 90, 20), -1)
    cv2.circle(image, (160, 90), 14, (20, 20, 230), -1)
    np.testing.assert_allclose(marker(image, 'blue'), [55, 90], atol=1)
    np.testing.assert_allclose(marker(image, 'red'), [160, 90], atol=1)
    assert marker(np.zeros_like(image), 'red') is None


def test_occlusion_is_not_bridged_and_jumps_are_excluded():
    state = np.tile([.04, .08, .08, .08], (8, 1))
    state[5:, :2] += .07
    ep = {'state': state, 'time': np.arange(8)*.1, 'valid': np.array([1, 1, 0, 1, 1, 1, 1, 1], bool)}
    assert transitions(ep)[3].tolist() == [0, 3, 5, 6]


def test_red_cap_is_detected_beside_a_large_brown_hand_and_oversized_red_distractor():
    image = np.full((600, 800, 3), 220, np.uint8)
    cv2.rectangle(image, (0, 220), (420, 590), (60, 85, 140), -1)
    cv2.rectangle(image, (0, 0), (420, 180), (20, 20, 220), -1)
    cv2.circle(image, (600, 350), 40, (25, 25, 185), -1)
    np.testing.assert_allclose(marker(image, 'red'), [600, 350], atol=1)


def test_compressed_magenta_red_cap_is_selected_over_red_brown_skin():
    hsv = np.full((480, 360, 3), [0, 0, 230], np.uint8)
    cv2.rectangle(hsv, (0, 0), (220, 280), (6, 150, 90), -1)
    cv2.circle(hsv, (80, 370), 25, (166, 180, 165), -1)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    np.testing.assert_allclose(marker(image, 'red'), [80, 370], atol=1)


def test_split_holds_out_whole_episodes_and_is_reproducible():
    episodes = [{'id': str(i)} for i in range(10)]
    splits = episode_split(episodes)
    groups = [set(e['id'] for e in s) for s in splits]
    assert [len(g) for g in groups] == [6, 2, 2]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sum(map(len, groups)) == 10
    assert splits == episode_split(episodes)
    with pytest.raises(ValueError, match='at least 5'):
        episode_split(episodes[:4])


def test_learned_dynamics_are_rotation_equivariant_and_zero_action_is_stationary():
    torch.manual_seed(4)
    model = ContactDynamics().eval()
    r = torch.tensor([[-.021, .001], [-.033, .007]])
    a = torch.tensor([[.006, .001], [.007, -.001]])
    theta = .72
    rotation = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float32)
    with torch.no_grad():
        np.testing.assert_allclose(model(r@rotation.T, a@rotation.T), model(r, a)@rotation.T, atol=1e-8)
        np.testing.assert_allclose(model(r, torch.zeros_like(a)), 0, atol=1e-10)
        assert model(r*10, a).norm().item() < 1e-6


def test_robot_moves_object_through_contact_and_idle_object_stays_still():
    env = PushEnv()
    try:
        state = env.reset()
        initial_puck = state[2:].copy()
        for _ in range(10):
            state = env.step([0, 0])
        np.testing.assert_allclose(state[2:], initial_puck, atol=1e-4)
        goal = np.array([.21, .105])
        for _ in range(90):
            state = env.step(geometric_action(state, goal))
        assert state[2]-initial_puck[0] > .05
        assert np.linalg.norm(state[2:]-goal) < .025
        assert abs(env.data.body('puck').xpos[2]-.009) < .005
    finally:
        env.close()


def test_extraction_of_encoded_video_and_missing_frames(tmp_path):
    path = tmp_path/'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    if not writer.isOpened():
        pytest.skip('Host lacks MP4 encoder')
    for i in range(60):
        image = np.full((480, 640, 3), 235, np.uint8)
        cv2.circle(image, (300+i, 250), 18, (20, 20, 230), -1)
        if not 21 <= i <= 29:
            cv2.circle(image, (270+i, 250), 10, (230, 90, 20), -1)
        writer.write(image)
    writer.release()
    frame = np.zeros((480, 640, 3), np.uint8)
    save_calibration(frame, [[30, 30], [610, 30], [610, 450], [30, 450]], tmp_path/'calibration.json')
    calibration = json.loads((tmp_path/'calibration.json').read_text())
    episode = extract_video(path, calibration, tmp_path/'preview.mp4')
    assert len(episode['state']) == 20
    assert not episode['valid'][7:10].any()
    assert episode['valid'][:7].all()
    assert not any(i in transitions(episode)[3] for i in [6, 7, 8, 9])


def test_circle_relative_coordinates_ignore_translation_and_uniform_zoom():
    points = np.array([[320., 510.], [540., 740.]])
    centers = np.array([[200., 800.], [215., 815.]])
    radii = np.array([130., 137.])
    theta = .7
    frame = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    original = task_coordinates(points, centers, radii, frame)
    shift, zoom = np.array([70., -40.]), 1.45
    transformed = task_coordinates(points*zoom+shift, centers*zoom+shift, radii*zoom, frame)
    np.testing.assert_allclose(original, transformed, atol=1e-10)


def test_drawn_circle_tracking_separates_target_from_coloured_cap():
    image = np.full((1300, 1000, 3), 235, np.uint8)
    cv2.circle(image, (280, 1040), 160, (25, 25, 25), 8)
    cv2.circle(image, (750, 300), 90, (20, 20, 180), -1)
    circle = target_circle(image)
    assert circle is not None
    np.testing.assert_allclose(circle[0], [280, 1040], atol=2)
    assert abs(circle[1]-160) < 7


def test_low_resolution_faint_cropped_circle_and_short_arc_rejection():
    image = np.full((480, 360, 3), 230, np.uint8)
    cv2.circle(image, (45, 360), 55, (115, 115, 115), 2)
    cv2.rectangle(image, (130, 30), (359, 290), (75, 95, 150), -1)
    cv2.circle(image, (300, 350), 24, (20, 20, 180), -1)
    circle = target_circle(image)
    assert circle is not None
    np.testing.assert_allclose(circle[0], [45, 360], atol=2)
    assert abs(circle[1]-55) < 3
    short_arc = np.full_like(image, 230)
    cv2.ellipse(short_arc, (100, 360), (55, 55), 0, 0, 150, (115, 115, 115), 2)
    assert target_circle(short_arc) is None
