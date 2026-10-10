"""Portable H.264 output for browser-playable tracking and simulation videos."""
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


class VideoWriter:
    def __init__(self, path, fps, size):
        self.size = size
        self.closed = False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
               '-i', '-', '-an', '-vcodec', 'libx264', '-preset', 'veryfast', '-crf', '22',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def isOpened(self):
        return not self.closed and self.process.poll() is None

    def write(self, frame):
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype != np.uint8:
            raise ValueError('Video frames must be BGR uint8 images of the configured dimensions')
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as e:
            details = self.process.stderr.read().decode(errors='replace')[:1000]
            raise RuntimeError(f'Video encoder failed: {details}') from e

    def release(self):
        if self.closed:
            return
        self.closed = True
        self.process.stdin.close()
        errors = self.process.stderr.read().decode(errors='replace')[:1000]
        code = self.process.wait()
        self.process.stderr.close()
        if code:
            raise RuntimeError(f'Video encoder exited with {code}: {errors}')
