"""Four-click, loopback-only browser calibration, with a CLI corner alternative."""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

from .data import write_json


def save_calibration(frame, corners, out):
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("Provide four finite pixel corners: top-left, top-right, bottom-right, bottom-left")
    if not cv2.isContourConvex(corners.astype(np.int32)) or abs(cv2.contourArea(corners)) < 1000:
        raise ValueError("Corners must form a convex quadrilateral in the requested order")
    if np.any(corners < 0) or np.any(corners[:, 0] >= frame.shape[1]) or np.any(corners[:, 1] >= frame.shape[0]):
        raise ValueError("Corners must be inside the image")
    destination = np.array([[0, 0.210], [0.297, 0.210], [0.297, 0], [0, 0]], np.float32)
    h = cv2.getPerspectiveTransform(corners, destination)
    write_json(out, {"corners_px": corners.tolist(), "homography": h.tolist(),
                     "image_size": [frame.shape[1], frame.shape[0]],
                     "paper_size_m": [0.297, 0.210], "sample_hz": 10,
                     "assumption": "fixed camera; A4 plane; marker height introduces parallax"})


PAGE = '''<!doctype html><html><meta charset="utf-8"><title>PushBridge calibration</title>
<style>body{font:17px system-ui;background:#101a27;color:#eee;max-width:1000px;margin:32px auto;padding:0 20px}canvas{max-width:100%;cursor:crosshair;border-radius:8px}button{padding:12px 20px;margin:12px 12px 0 0;font:inherit}#status{color:#94d7c1}</style>
<h1>Mark the A4 sheet</h1><p>Click its four corners in order: <b>top-left → top-right → bottom-right → bottom-left</b>. The phone must remain fixed for all recordings.</p>
<canvas id="c"></canvas><div><button id="undo">Undo</button><button id="save" disabled>Save calibration</button></div><p id="status">Click top-left.</p>
<script>const c=document.querySelector('#c'),x=c.getContext('2d'),im=new Image(),p=[];
const names=['top-left','top-right','bottom-right','bottom-left'];
function draw(){x.drawImage(im,0,0);p.forEach((v,i)=>{x.fillStyle='#ffdb6e';x.beginPath();x.arc(v[0],v[1],10,0,7);x.fill();x.font='28px sans-serif';x.fillText(i+1,v[0]+14,v[1]);});document.querySelector('#save').disabled=p.length!==4;document.querySelector('#status').textContent=p.length===4?'Check the order, then save.':'Click '+names[p.length]+'.';}
im.onload=()=>{c.width=im.width;c.height=im.height;draw()};im.src='/frame.jpg';
c.onclick=e=>{if(p.length<4){let b=c.getBoundingClientRect();p.push([(e.clientX-b.left)*c.width/b.width,(e.clientY-b.top)*c.height/b.height]);draw();}};
document.querySelector('#undo').onclick=()=>{p.pop();draw()};
document.querySelector('#save').onclick=async()=>{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corners:p})});const j=await r.json();document.querySelector('#status').textContent=j.message;};</script></html>'''


def calibrate(video, out, corners=None, port=8765):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read video: {video}")
    if corners is not None:
        save_calibration(frame, corners, out)
        print(f"Saved {out}")
        return
    encoded = cv2.imencode('.jpg', frame)[1].tobytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path not in ['/', '/frame.jpg']:
                self.send_error(404)
                return
            body = PAGE.encode() if self.path == '/' else encoded
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if self.path == '/' else 'image/jpeg')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != '/save':
                self.send_error(404)
                return
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{port}':
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length > 4096:
                    raise ValueError('Request too large')
                value = json.loads(self.rfile.read(length))
                save_calibration(frame, value['corners'], out)
                message, status = f'Saved {Path(out).name}. You may close this tab.', 200
            except (ValueError, KeyError) as e:
                message, status = str(e), 400
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'message': message}).encode())
            if status == 200:
                threading.Thread(target=self.server.shutdown, daemon=True).start()
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'Open http://127.0.0.1:{port} and click the four paper corners. Ctrl+C cancels.', flush=True)
    webbrowser.open(f'http://127.0.0.1:{port}')
    try:
        server.serve_forever()
    finally:
        server.server_close()
