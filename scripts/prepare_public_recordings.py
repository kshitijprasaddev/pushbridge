"""Remove location, device metadata and audio from publication copies.

Original videos stay in the explicitly supplied directory. Compressed video
packets are copied; a full decoded-frame hash verifies identical image pixels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import cv2

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_sha(path):
    result = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-map', '0:v:0',
                             '-an', '-f', 'hash', '-hash', 'sha256', '-'],
                            check=True, text=True, capture_output=True)
    return result.stdout.strip().split('=', 1)[1]


def metadata_keys(path):
    result = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-f', 'ffmetadata', '-'],
                            check=True, text=True, capture_output=True)
    return [line.split('=', 1)[0] for line in result.stdout.splitlines() if '=' in line]


def shape(path):
    cap = cv2.VideoCapture(str(path))
    values = [cap.get(key) for key in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS,
                                      cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    cap.release()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='Directory containing originals, outside the public repository')
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    if source_dir == ROOT or ROOT in source_dir.parents:
        raise ValueError('Keep originals outside the publication repository')
    manifest_path = ROOT/'data/processed/personal_five/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = []
    for episode in manifest['episodes']:
        source = source_dir/episode['raw_name']
        expected = episode.get('original_raw_sha256', episode['raw_sha256'])
        if sha(source) != expected:
            raise ValueError(f'Original differs from extraction source: {source.name}')
        target = ROOT/'data/raw/personal_circle'/source.name
        temp = target.with_name(f'.{source.stem}.publication.mov')
        subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(source), '-map', '0:v:0',
                        '-c:v', 'copy', '-map_metadata', '-1', '-an', '-fflags', '+bitexact',
                        '-movflags', '+faststart', str(temp)], check=True)
        original_pixels, public_pixels = decoded_sha(source), decoded_sha(temp)
        if original_pixels != public_pixels or shape(source) != shape(temp):
            raise ValueError(f'Video pixels or sample geometry changed: {source.name}')
        keys = metadata_keys(temp)
        if any('location' in key.lower() or 'gps' in key.lower() for key in keys):
            raise ValueError(f'Location metadata remains: {source.name}')
        temp.replace(target)
        episode['original_raw_sha256'] = expected
        episode['raw_sha256'] = sha(target)
        episode['publication_copy'] = {
            'video_packets': 'Copied without video re-encoding',
            'decoded_video_sha256': public_pixels,
            'original_decoded_video_sha256': original_pixels,
            'audio': 'Removed', 'location_and_device_metadata': 'Removed',
        }
        records.append({'file': source.name, 'original_sha256': expected,
                        'publication_sha256': episode['raw_sha256'],
                        'decoded_video_sha256': public_pixels,
                        'original_decoded_video_sha256': original_pixels,
                        'metadata_keys': keys, 'frame_count_fps_width_height': shape(target)})
        print(f'Prepared {source.name}: all decoded video frames unchanged; location metadata removed.', flush=True)
    manifest['publication_note'] = 'Published raw files preserve every decoded video frame and sample geometry. Audio and location/device metadata were removed; originals remain outside the public repository.'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    record = {'status': 'passed', 'method': 'Lossless compressed-video remux; full decoded-frame SHA-256 comparison',
              'location_metadata_removed': True, 'audio_removed': True,
              'originals_preserved_outside_repository': True, 'files': records}
    (ROOT/'data/raw/personal_circle/publication_verification.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
