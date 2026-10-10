import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { LINKS, worldPosition, sampleTrace } from "./kinematics.mjs";

// This scene renders recorded MuJoCo poses. It does not compute contacts or actions.
export default function RobotScene({ trace, position, view, showPath }) {
  const host = useRef(null),
    engine = useRef(null);
  const live = useRef({ trace, position, view, showPath });
  live.current = { trace, position, view, showPath };
  const [error, setError] = useState("");
  useEffect(() => {
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
      });
    } catch {
      setError(
        "WebGL is unavailable. Open the native physics replay below to view this task.",
      );
      return;
    }
    const el = host.current;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);
    renderer.domElement.setAttribute(
      "aria-label",
      "Three-dimensional replay of the compact robot and recorded object motion",
    );
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#ced2cf");
    const pmrem = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    const env = pmrem.fromScene(room, 0.04);
    scene.environment = env.texture;
    scene.environmentIntensity = 0.65;
    room.dispose();
    pmrem.dispose();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.001, 12);
    camera.position.set(0.83, 0.58, 0.72);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.255, 0.035, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.065;
    controls.maxDistance = 1.7;
    controls.maxPolarAngle = Math.PI * 0.485;
    controls.enablePan = false;
    const materials = [];
    const mat = (color, metalness = 0, roughness = 0.4, extra = {}) => {
      const m = new THREE.MeshPhysicalMaterial({
        color,
        metalness,
        roughness,
        ...extra,
      });
      materials.push(m);
      return m;
    };
    const alloy = mat("#c0c7c9", 0.85, 0.29),
      bright = mat("#e5e9e6", 0.7, 0.24);
    const black = mat("#242a2d", 0.4, 0.34),
      graphite = mat("#3a4143", 0.64, 0.36);
    const lime = mat("#d8f17e", 0.12, 0.3),
      green = mat("#37532a", 0, 1);
    const red = mat("#d84d3e", 0.28, 0.26, {
      clearcoat: 0.4,
      clearcoatRoughness: 0.24,
    });
    const tableMat = mat("#707979", 0.7, 0.42),
      white = mat("#e2e6df", 0.12, 0.75);
    function mesh(geo, material, parent = scene, pos = [0, 0, 0]) {
      const m = new THREE.Mesh(geo, material);
      m.position.set(...pos);
      m.castShadow = true;
      m.receiveShadow = true;
      parent.add(m);
      return m;
    }
    function box(size, material, parent, pos, radius = 0.003) {
      return mesh(
        new RoundedBoxGeometry(...size, 3, radius),
        material,
        parent,
        pos,
      );
    }
    function cylinder(
      radius,
      height,
      material,
      parent,
      pos,
      radiusTop = radius,
    ) {
      return mesh(
        new THREE.CylinderGeometry(radiusTop, radius, height, 48),
        material,
        parent,
        pos,
      );
    }
    function ring(inner, outer, material, parent, pos) {
      const m = mesh(
        new THREE.RingGeometry(inner, outer, 80),
        material,
        parent,
        pos,
      );
      m.rotation.x = -Math.PI / 2;
      return m;
    }
    const ambient = new THREE.HemisphereLight("#f4f8ef", "#5b6875", 1.35);
    scene.add(ambient);
    const key = new THREE.DirectionalLight("#fff8e9", 2.7);
    key.position.set(0.2, 1.1, 0.35);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -0.75;
    key.shadow.camera.right = 0.75;
    key.shadow.camera.top = 0.65;
    key.shadow.camera.bottom = -0.65;
    key.shadow.camera.near = 0.01;
    key.shadow.camera.far = 3;
    key.shadow.bias = -0.00008;
    key.shadow.normalBias = 0.0003;
    key.shadow.radius = 4;
    key.target.position.set(0.25, 0, 0);
    scene.add(key, key.target);
    const fill = new THREE.DirectionalLight("#e1edff", 1.2);
    fill.position.set(-0.4, 0.45, -0.6);
    scene.add(fill);
    const floor = mesh(
      new THREE.PlaneGeometry(200, 200),
      mat("#c2c8c5", 0.05, 0.85),
      scene,
      [0, -0.084, 0],
    );
    floor.rotation.x = -Math.PI / 2;
    box([0.82, 0.065, 0.49], tableMat, scene, [0.255, -0.039, 0], 0.013);
    box([0.79, 0.01, 0.46], alloy, scene, [0.255, -0.0035, 0], 0.006);
    // Recesses and fasteners are presentation details. Joint centres and task sizes are exact.
    for (const x of [-0.1, 0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.61])
      for (const z of [-0.195, 0.195]) {
        cylinder(0.0031, 0.001, black, scene, [x, 0.0021, z]);
        cylinder(0.001, 0.0011, graphite, scene, [x, 0.0028, z]);
      }
    const task = new THREE.Group();
    scene.add(task);
    box([0.297, 0.0007, 0.21], white, task, [0.3985, 0.001, 0], 0.001);
    const grid = new THREE.Group();
    scene.add(grid);
    const gridmat = new THREE.LineBasicMaterial({
      color: "#778582",
      transparent: true,
      opacity: 0.17,
    });
    materials.push(gridmat);
    for (let x = 0.26; x < 0.548; x += 0.01) {
      const geom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, 0.0018, -0.1),
        new THREE.Vector3(x, 0.0018, 0.1),
      ]);
      grid.add(new THREE.Line(geom, gridmat));
    }
    for (let z = -0.1; z <= 0.1; z += 0.01) {
      const geom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0.255, 0.0018, z),
        new THREE.Vector3(0.542, 0.0018, z),
      ]);
      grid.add(new THREE.Line(geom, gridmat));
    }
    // Procedural printed labels remain crisp and require no remote fonts or textures.
    const textureCanvases = [];
    function label(text, width, position, color = "#566360", font = 32) {
      const c = document.createElement("canvas");
      c.width = 1024;
      c.height = 128;
      const ctx = c.getContext("2d");
      ctx.font = `${font}px monospace`;
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.fillText(text, 512, 75);
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace;
      textureCanvases.push(tex);
      const m = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        depthWrite: false,
      });
      materials.push(m);
      const p = mesh(
        new THREE.PlaneGeometry(width, width / 8),
        m,
        scene,
        position,
      );
      p.rotation.x = -Math.PI / 2;
      p.castShadow = false;
    }
    label("PUSHBRIDGE  /  CONTACT TRANSFER LAB", 0.245, [0.395, 0.0022, 0.094]);
    label("COMPACT 3R  |  0.25 RAD/S", 0.15, [0.074, 0.0023, 0.179]);
    const base = new THREE.Group();
    scene.add(base);
    box([0.12, 0.012, 0.12], graphite, base, [0, 0.008, 0], 0.009);
    for (const x of [-0.049, 0.049])
      for (const z of [-0.049, 0.049])
        cylinder(0.0035, 0.003, alloy, base, [x, 0.016, z]);
    cylinder(0.048, 0.018, black, base, [0, 0.024, 0]);
    cylinder(0.042, 0.085, alloy, base, [0, 0.075, 0]);
    cylinder(0.043, 0.009, graphite, base, [0, 0.12, 0]);
    ring(0.041, 0.043, lime, base, [0, 0.125, 0]);
    const pivots = [],
      root = new THREE.Group();
    root.position.y = 0.15;
    scene.add(root);
    let parent = root;
    LINKS.forEach((length, i) => {
      const pivot = new THREE.Group();
      parent.add(pivot);
      pivots.push(pivot);
      const r = [0.035, 0.026, 0.02][i];
      cylinder(r, 0.048 - i * 0.007, black, pivot, [0, 0, 0]);
      cylinder(r * 0.92, 0.009, bright, pivot, [0, 0.027 - i * 0.003, 0]);
      cylinder(r * 0.55, 0.006, graphite, pivot, [0, 0.033 - i * 0.003, 0]);
      for (let j = 0; j < 6; j++) {
        const a = (j * Math.PI) / 3;
        cylinder(0.0018, 0.002, black, pivot, [
          Math.cos(a) * r * 0.74,
          0.033 - i * 0.003,
          Math.sin(a) * r * 0.74,
        ]);
      }
      const w = [0.035, 0.029, 0.024][i];
      box(
        [length - r * 0.8, w, w * 0.94],
        i === 2 ? lime : alloy,
        pivot,
        [length / 2 + r * 0.2, 0, 0],
        0.007,
      );
      box(
        [length - r * 1.6, 0.004, w * 0.78],
        i === 2 ? graphite : bright,
        pivot,
        [length / 2 + r * 0.22, w / 2 + 0.001, 0],
        0.0015,
      );
      box(
        [length * 0.5, 0.001, 0.003],
        black,
        pivot,
        [length * 0.54, w / 2 + 0.0035, -0.006],
        0.0004,
      );
      const next = new THREE.Group();
      next.position.x = length;
      pivot.add(next);
      parent = next;
    });
    cylinder(0.009, 0.018, black, parent, [0, -0.004, 0]);
    cylinder(0.005, 0.115, alloy, parent, [0, -0.071, 0]);
    cylinder(0.0045, 0.009, graphite, parent, [0, -0.13, 0]);
    const tip = mesh(
      new THREE.SphereGeometry(0.003, 24, 16),
      lime,
      parent,
      [0, -0.14, 0],
    );
    const puck = new THREE.Group();
    scene.add(puck);
    const outline = [
      new THREE.Vector2(0, 0),
      new THREE.Vector2(0.011, 0),
      new THREE.Vector2(0.012, 0.001),
      new THREE.Vector2(0.012, 0.017),
      new THREE.Vector2(0.011, 0.018),
      new THREE.Vector2(0, 0.018),
    ];
    mesh(new THREE.LatheGeometry(outline, 64), red, puck);
    cylinder(0.0088, 0.0002, red, puck, [0, 0.0181, 0]);
    ring(0.0086, 0.0089, mat("#e77e62", 0.35, 0.4), puck, [0, 0.0183, 0]);
    cylinder(0.001, 0.0002, white, puck, [0, 0.0184, 0]);
    const goal = new THREE.Group();
    scene.add(goal);
    const goalMat = mat("#72a839", 0, 1);
    const goalDisk = mesh(
      new THREE.CircleGeometry(0.02, 80),
      goalMat,
      goal,
      [0, 0.002, 0],
    );
    goalDisk.rotation.x = -Math.PI / 2;
    goalDisk.castShadow = false;
    ring(0.01955, 0.02, green, goal, [0, 0.00215, 0]);
    const safeMat = new THREE.LineBasicMaterial({
      color: "#577041",
      transparent: true,
      opacity: 0.6,
    });
    materials.push(safeMat);
    const safeCircle = [];
    for (let i = 0; i < 65; i++)
      safeCircle.push(
        new THREE.Vector3(
          0.008 * Math.cos((i * Math.PI) / 32),
          0.0022,
          0.008 * Math.sin((i * Math.PI) / 32),
        ),
      );
    goal.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(safeCircle),
        safeMat,
      ),
    );
    const path = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: "#668844",
        transparent: true,
        opacity: 0.65,
      }),
    );
    scene.add(path);
    materials.push(path.material);
    let previousTrace = null,
      previousView = null,
      frame,
      visible = true,
      interrupted = false,
      dirty = true,
      renderedPosition = -1,
      renderedPath = null;
    const camTarget = new THREE.Vector3(),
      desiredPos = new THREE.Vector3(),
      desiredTarget = new THREE.Vector3();
    renderer.domElement.addEventListener("webglcontextlost", onLost);
    function onLost(e) {
      e.preventDefault();
      setError(
        "The 3D graphics context was interrupted. Reload the viewer or open the native physics replay.",
      );
    }
    controls.addEventListener("start", () => {
      interrupted = true;
    });
    const resize = new ResizeObserver(() => {
      const { width, height } = el.getBoundingClientRect();
      if (width && height) {
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        dirty = true;
      }
    });
    resize.observe(el);
    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    observer.observe(el);
    function animate() {
      frame = requestAnimationFrame(animate);
      if (!visible || document.hidden) return;
      const value = live.current,
        t = value.trace;
      const traceChanged = t !== previousTrace;
      if (
        traceChanged ||
        value.position !== renderedPosition ||
        value.showPath !== renderedPath
      )
        dirty = true;
      if (t) {
        const sample = sampleTrace(t, value.position);
        pivots.forEach((p, i) => {
          p.rotation.y = sample.joint[i];
        });
        puck.position.set(
          ...worldPosition(sample.state[2], sample.state[3], 0.0015),
        );
        goal.position.set(...worldPosition(t.goal[0], t.goal[1]));
        if (t !== previousTrace) {
          path.geometry.dispose();
          path.geometry = new THREE.BufferGeometry().setFromPoints(
            t.state.map(
              (s) => new THREE.Vector3(...worldPosition(s[2], s[3], 0.0024)),
            ),
          );
          previousTrace = t;
        }
        path.geometry.setDrawRange(0, Math.max(0, sample.index + 1));
        path.visible = value.showPath;
        if (
          value.view !== previousView ||
          (traceChanged && value.view === "detail")
        ) {
          interrupted = false;
          if (value.view === "top") {
            const topDistance = Math.max(
              0.75,
              0.74 /
                (2 *
                  Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) *
                  camera.aspect),
            );
            desiredPos.set(0.255, topDistance, 0.02);
            desiredTarget.set(0.255, 0.01, 0);
          } else if (value.view === "detail") {
            camTarget.set(...worldPosition(t.goal[0], t.goal[1]));
            desiredTarget.copy(camTarget);
            desiredPos
              .copy(camTarget)
              .add(new THREE.Vector3(0.065, 0.12, 0.14));
          } else {
            desiredPos.set(0.83, 0.58, 0.72);
            desiredTarget.set(0.255, 0.035, 0);
          }
          previousView = value.view;
        }
        if (!interrupted) {
          const movingCamera =
            camera.position.distanceToSquared(desiredPos) > 1e-10 ||
            controls.target.distanceToSquared(desiredTarget) > 1e-10;
          if (movingCamera) {
            camera.position.lerp(desiredPos, 0.085);
            controls.target.lerp(desiredTarget, 0.085);
            dirty = true;
          } else {
            camera.position.copy(desiredPos);
            controls.target.copy(desiredTarget);
          }
        }
      }
      const orbitChanged = controls.update();
      if (dirty || orbitChanged) {
        renderer.render(scene, camera);
        renderedPosition = value.position;
        renderedPath = value.showPath;
        dirty = false;
      }
    }
    animate();
    engine.current = { renderer, scene, camera };
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      observer.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("webglcontextlost", onLost);
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
      });
      materials.forEach((m) => m.dispose());
      textureCanvases.forEach((t) => t.dispose());
      env.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      engine.current = null;
    };
  }, []);
  return (
    <div className="robot-scene" ref={host}>
      {error && (
        <div className="scene-error" role="alert">
          {error}
        </div>
      )}
    </div>
  );
}
