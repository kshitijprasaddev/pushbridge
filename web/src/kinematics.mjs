// MuJoCo world coordinates (x, y, z) map to Three.js (x, z, -y).
export const LINKS = [0.24, 0.17, 0.13];
export const ORIGIN = [0.25, -0.105];
export const FIT_MM = 8;
export function forwardKinematics(q) {
  let x = 0,
    y = 0,
    a = 0;
  const points = [[0, 0]];
  LINKS.forEach((length, i) => {
    a += q[i];
    x += length * Math.cos(a);
    y += length * Math.sin(a);
    points.push([x, y]);
  });
  return points;
}
export function worldPosition(x, y, height = 0) {
  return [x + ORIGIN[0], height, -(y + ORIGIN[1])];
}
export function distanceMM(state, goal) {
  return Math.hypot(state[2] - goal[0], state[3] - goal[1]) * 1000;
}
export function sampleTrace(trace, position) {
  const lo = Math.min(trace.state.length - 1, Math.floor(position));
  const hi = Math.min(trace.state.length - 1, lo + 1),
    t = position - lo;
  const mix = (a, b) => a.map((v, i) => v + (b[i] - v) * t);
  return {
    state: mix(trace.state[lo], trace.state[hi]),
    joint: mix(trace.joint[lo], trace.joint[hi]),
    index: lo,
  };
}
