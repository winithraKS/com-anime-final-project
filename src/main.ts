import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

// ─────────────────────────────────────────────────────────────────────────────
// Scene setup  (same as your original)
// ─────────────────────────────────────────────────────────────────────────────

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);
const canvas = document.querySelector<HTMLCanvasElement>("#bg");
if (!canvas) throw new Error('Canvas element "#bg" not found');

const renderer = new THREE.WebGLRenderer({ canvas });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
camera.position.set(0, 10, -35);

const controls = new OrbitControls(camera, renderer.domElement);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
scene.add(directionalLight);
scene.add(new THREE.GridHelper(200, 50));

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type Vec3 = [number, number, number];
type Face = [number, number, number];

interface SimplifyResult {
  vertices: Float32Array;   // (N*3) compacted vertex positions
  faces: Uint32Array;        // (M*3) triangle indices
  /** For each compacted vertex index → original vertex index before compaction */
  compactToOriginal: Int32Array;
}

// ─────────────────────────────────────────────────────────────────────────────
// QEM simplification  (JavaScript port of the Python implementation)
// ─────────────────────────────────────────────────────────────────────────────

/** Build per-vertex quadric matrices from face plane equations. */
function computeQuadrics(
  verts: Float64Array,
  faces: Int32Array,
  nVerts: number,
): Float64Array {
  // Q is stored flat: nVerts × 16 (row-major 4×4)
  const Q = new Float64Array(nVerts * 16);

  for (let fi = 0; fi < faces.length / 3; fi++) {
    const i0 = faces[fi * 3];
    const i1 = faces[fi * 3 + 1];
    const i2 = faces[fi * 3 + 2];

    const ax = verts[i0 * 3], ay = verts[i0 * 3 + 1], az = verts[i0 * 3 + 2];
    const bx = verts[i1 * 3], by = verts[i1 * 3 + 1], bz = verts[i1 * 3 + 2];
    const cx = verts[i2 * 3], cy = verts[i2 * 3 + 1], cz = verts[i2 * 3 + 2];

    // Edge vectors
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;

    // Normal = cross(e1, e2)
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-12) continue;
    nx /= len; ny /= len; nz /= len;

    const d = -(nx * ax + ny * ay + nz * az);
    // Plane vector p = [nx, ny, nz, d]
    // Kp = outer(p, p)  — 4x4
    const p = [nx, ny, nz, d];
    for (const vi of [i0, i1, i2]) {
      const base = vi * 16;
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
          Q[base + r * 4 + c] += p[r] * p[c];
    }
  }
  return Q;
}

/** Evaluate v^T Q v where v = [x,y,z,1] and Q is stored at Q[base..base+16]. */
function quadricCost(
  Q: Float64Array, base: number,
  x: number, y: number, z: number,
): number {
  // Qv = Q * [x,y,z,1]
  const v = [x, y, z, 1.0];
  let cost = 0;
  for (let r = 0; r < 4; r++) {
    let row = 0;
    for (let c = 0; c < 4; c++) row += Q[base + r * 4 + c] * v[c];
    cost += row * v[r];
  }
  return cost;
}

/** Attempt to solve 3×3 linear system Ax=b (in place, returns false if singular). */
function solve3(A: number[], b: number[], out: number[]): boolean {
  // Gaussian elimination with partial pivoting
  const M: number[][] = [
    [A[0], A[1], A[2], b[0]],
    [A[3], A[4], A[5], b[1]],
    [A[6], A[7], A[8], b[2]],
  ];
  for (let col = 0; col < 3; col++) {
    // Find pivot
    let maxVal = Math.abs(M[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    if (maxVal < 1e-10) return false;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    const inv = 1 / M[col][col];
    for (let row = col + 1; row < 3; row++) {
      const factor = M[row][col] * inv;
      for (let k = col; k <= 3; k++) M[row][k] -= factor * M[col][k];
    }
  }
  // Back substitution
  for (let row = 2; row >= 0; row--) {
    let s = M[row][3];
    for (let k = row + 1; k < 3; k++) s -= M[row][k] * out[k];
    out[row] = s / M[row][row];
  }
  return true;
}

/** Find the optimal collapse target given the combined quadric Qsum (flat 16-element array). */
function optimalVertex(
  Qsum: Float64Array | number[],
  base: number,
  vi_x: number, vi_y: number, vi_z: number,
  vj_x: number, vj_y: number, vj_z: number,
): Vec3 {
  // Build 3×3 submatrix A = Qsum[0:3, 0:3], b = -Qsum[0:3, 3]
  const A = [
    Qsum[base + 0], Qsum[base + 1], Qsum[base + 2],
    Qsum[base + 4], Qsum[base + 5], Qsum[base + 6],
    Qsum[base + 8], Qsum[base + 9], Qsum[base + 10],
  ];
  const b = [-Qsum[base + 3], -Qsum[base + 7], -Qsum[base + 11]];
  const out = [0, 0, 0];
  if (solve3(A, b, out)) return out as Vec3;
  // fallback: midpoint
  return [
    (vi_x + vj_x) * 0.5,
    (vi_y + vj_y) * 0.5,
    (vi_z + vj_z) * 0.5,
  ];
}

/** Simple min-heap keyed on a number. */
class MinHeap<T> {
  private data: { key: number; val: T }[] = [];
  push(key: number, val: T) {
    this.data.push({ key, val });
    this._bubbleUp(this.data.length - 1);
  }
  pop(): { key: number; val: T } | undefined {
    if (!this.data.length) return undefined;
    const top = this.data[0];
    const last = this.data.pop()!;
    if (this.data.length) { this.data[0] = last; this._sinkDown(0); }
    return top;
  }
  get size() { return this.data.length; }
  private _bubbleUp(i: number) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p].key <= this.data[i].key) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }
  private _sinkDown(i: number) {
    const n = this.data.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.data[l].key < this.data[smallest].key) smallest = l;
      if (r < n && this.data[r].key < this.data[smallest].key) smallest = r;
      if (smallest === i) break;
      [this.data[smallest], this.data[i]] = [this.data[i], this.data[smallest]];
      i = smallest;
    }
  }
}

/**
 * QEM mesh simplification.
 *
 * @param vertsIn  Float64Array, length = nVerts * 3
 * @param facesIn  Int32Array,   length = nFaces * 3
 * @param ratio    target face ratio, e.g. 0.1 = keep 10% of faces
 * @param onProgress  optional callback(fraction 0..1)
 */
export function qemSimplify(
  vertsIn: Float64Array,
  facesIn: Int32Array,
  ratio: number,
  onProgress?: (f: number) => void,
): SimplifyResult {
  const nVertsOrig = vertsIn.length / 3;
  const nFacesOrig = facesIn.length / 3;
  const targetFaces = Math.max(1, Math.floor(nFacesOrig * ratio));

  // Working copies
  const pos = new Float64Array(vertsIn);              // current positions (by root)
  const Q = computeQuadrics(pos, facesIn, nVertsOrig); // Q[vi * 16 .. +16]

  // Union-Find
  const parent = Int32Array.from({ length: nVertsOrig }, (_, i) => i);
  function find(x: number): number {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  function union(a: number, b: number, nx: number, ny: number, nz: number) {
    // merge b into a
    parent[b] = a;
    pos[a * 3] = nx; pos[a * 3 + 1] = ny; pos[a * 3 + 2] = nz;
    // accumulate quadrics
    for (let k = 0; k < 16; k++) Q[a * 16 + k] += Q[b * 16 + k];
  }

  // Build face list as flat Int32Array; -1 = deleted
  const faces = new Int32Array(facesIn);   // mutable copy

  // Collect unique edges
  const edgeSet = new Set<number>();
  for (let fi = 0; fi < nFacesOrig; fi++) {
    const a = faces[fi * 3], b = faces[fi * 3 + 1], c = faces[fi * 3 + 2];
    const encode = (u: number, v: number) =>
      u < v ? u * nVertsOrig + v : v * nVertsOrig + u;
    edgeSet.add(encode(a, b));
    edgeSet.add(encode(b, c));
    edgeSet.add(encode(a, c));
  }

  // Build heap
  const heap = new MinHeap<[number, number, Vec3]>();
  for (const code of edgeSet) {
    const vi = Math.floor(code / nVertsOrig);
    const vj = code % nVertsOrig;

    // Combined quadric stored temporarily in a local array
    const Qc = new Array(16);
    for (let k = 0; k < 16; k++) Qc[k] = Q[vi * 16 + k] + Q[vj * 16 + k];

    const opt = optimalVertex(
      Qc, 0,
      pos[vi * 3], pos[vi * 3 + 1], pos[vi * 3 + 2],
      pos[vj * 3], pos[vj * 3 + 1], pos[vj * 3 + 2],
    );
    // Evaluate cost using combined Q
    let cost = 0;
    const v4 = [...opt, 1.0];
    for (let r = 0; r < 4; r++) {
      let row = 0;
      for (let c2 = 0; c2 < 4; c2++) row += Qc[r * 4 + c2] * v4[c2];
      cost += row * v4[r];
    }
    heap.push(cost, [vi, vj, opt]);
  }

  let nFaces = nFacesOrig;
  let collapses = 0;
  const totalCollapses = nFacesOrig - targetFaces;

  while (nFaces > targetFaces && heap.size > 0) {
    const item = heap.pop()!;
    const [vi, vj, vopt] = item.val;
    const ri = find(vi), rj = find(vj);
    if (ri === rj) continue;  // already merged

    // Collapse rj into ri
    union(ri, rj, vopt[0], vopt[1], vopt[2]);

    // Update faces, count degenerates
    for (let fi = 0; fi < nFacesOrig; fi++) {
      if (faces[fi * 3] === -1) continue;
      let changed = false;
      for (let k = 0; k < 3; k++) {
        const old = faces[fi * 3 + k];
        const nr = find(old);
        if (nr !== old) { faces[fi * 3 + k] = nr; changed = true; }
      }
      if (changed) {
        const a = faces[fi * 3], b = faces[fi * 3 + 1], c = faces[fi * 3 + 2];
        if (a === b || b === c || a === c) {
          faces[fi * 3] = -1;  // mark deleted
          nFaces--;
        }
      }
    }

    // Re-add edges touching ri
    const neighbors = new Set<number>();
    for (let fi = 0; fi < nFacesOrig; fi++) {
      if (faces[fi * 3] === -1) continue;
      const a = faces[fi * 3], b = faces[fi * 3 + 1], c = faces[fi * 3 + 2];
      if (a === ri || b === ri || c === ri) {
        if (a !== ri) neighbors.add(a);
        if (b !== ri) neighbors.add(b);
        if (c !== ri) neighbors.add(c);
      }
    }
    for (const nb of neighbors) {
      const Qc = new Array(16);
      for (let k = 0; k < 16; k++) Qc[k] = Q[ri * 16 + k] + Q[nb * 16 + k];
      const opt2 = optimalVertex(
        Qc, 0,
        pos[ri * 3], pos[ri * 3 + 1], pos[ri * 3 + 2],
        pos[nb * 3], pos[nb * 3 + 1], pos[nb * 3 + 2],
      );
      let cost2 = 0;
      const v4 = [...opt2, 1.0];
      for (let r = 0; r < 4; r++) {
        let row = 0;
        for (let c2 = 0; c2 < 4; c2++) row += Qc[r * 4 + c2] * v4[c2];
        cost2 += row * v4[r];
      }
      heap.push(cost2, [ri, nb, opt2]);
    }

    collapses++;
    if (onProgress && collapses % 500 === 0) {
      onProgress(Math.min(collapses / totalCollapses, 1));
    }
  }

  // ── Compact ──────────────────────────────────────────────────────────────
  // Collect valid faces and remap vertices
  const validFaces: number[] = [];
  for (let fi = 0; fi < nFacesOrig; fi++) {
    if (faces[fi * 3] !== -1) {
      validFaces.push(faces[fi * 3], faces[fi * 3 + 1], faces[fi * 3 + 2]);
    }
  }

  // Collect unique roots used
  const usedRoots = new Set<number>(validFaces);
  const sortedRoots = Array.from(usedRoots).sort((a, b) => a - b);
  const rootToCompact = new Map<number, number>();
  sortedRoots.forEach((r, i) => rootToCompact.set(r, i));

  const nCompact = sortedRoots.length;
  const compactVerts = new Float32Array(nCompact * 3);
  const compactToOriginal = new Int32Array(nCompact);

  for (let i = 0; i < nCompact; i++) {
    const r = sortedRoots[i];
    compactVerts[i * 3]     = pos[r * 3];
    compactVerts[i * 3 + 1] = pos[r * 3 + 1];
    compactVerts[i * 3 + 2] = pos[r * 3 + 2];
    compactToOriginal[i] = r;   // root index = original vertex index
  }

  const compactFaces = new Uint32Array(validFaces.length);
  for (let i = 0; i < validFaces.length; i++) {
    compactFaces[i] = rootToCompact.get(validFaces[i])!;
  }

  onProgress?.(1);
  return { vertices: compactVerts, faces: compactFaces, compactToOriginal };
}

// ─────────────────────────────────────────────────────────────────────────────
// KD-tree  (simple 3D, for displacement transfer)
// ─────────────────────────────────────────────────────────────────────────────

interface KDNode {
  idx: number;       // index into original vertex array
  axis: number;
  left?: KDNode;
  right?: KDNode;
}

function buildKDTree(indices: number[], verts: Float64Array, depth = 0): KDNode | undefined {
  if (!indices.length) return undefined;
  const axis = depth % 3;
  indices.sort((a, b) => verts[a * 3 + axis] - verts[b * 3 + axis]);
  const mid = indices.length >> 1;
  return {
    idx: indices[mid],
    axis,
    left: buildKDTree(indices.slice(0, mid), verts, depth + 1),
    right: buildKDTree(indices.slice(mid + 1), verts, depth + 1),
  };
}

function kdNearest(
  node: KDNode | undefined,
  verts: Float64Array,
  qx: number, qy: number, qz: number,
  best: { dist: number; idx: number },
) {
  if (!node) return;
  const px = verts[node.idx * 3], py = verts[node.idx * 3 + 1], pz = verts[node.idx * 3 + 2];
  const d = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2;
  if (d < best.dist) { best.dist = d; best.idx = node.idx; }

  const diff = [qx, qy, qz][node.axis] - [px, py, pz][node.axis];
  const near = diff < 0 ? node.left : node.right;
  const far  = diff < 0 ? node.right : node.left;
  kdNearest(near, verts, qx, qy, qz, best);
  if (diff * diff < best.dist) kdNearest(far, verts, qx, qy, qz, best);
}

// ─────────────────────────────────────────────────────────────────────────────
// Displacement transfer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * For each simplified vertex, find the nearest original vertex and
 * copy that vertex's (smile − neutral) displacement.
 *
 * origNeutral: Float64Array (nOrig * 3)
 * origSmile:   Float64Array (nOrig * 3)
 * simpVerts:   Float32Array (nSimp * 3)
 * compactToOriginal: Int32Array — direct index map from QEM (preferred)
 *
 * Returns Float32Array (nSimp * 3) of displacement vectors.
 */
function transferDisplacement(
  origNeutral: Float64Array,
  origSmile: Float64Array,
  simpVerts: Float32Array,
  compactToOriginal: Int32Array,
): Float32Array {
  const nOrig = origNeutral.length / 3;
  const nSimp = simpVerts.length / 3;

  // Per-vertex displacement on original mesh
  const origDisp = new Float32Array(nOrig * 3);
  for (let i = 0; i < nOrig * 3; i++) origDisp[i] = origSmile[i] - origNeutral[i];

  // Check if direct map is usable (all indices in range)
  const canUseDirect = compactToOriginal.every(idx => idx < nOrig);

  if (canUseDirect) {
    // Fast path: each compact vert's root IS an original vertex index
    const disp = new Float32Array(nSimp * 3);
    for (let i = 0; i < nSimp; i++) {
      const orig = compactToOriginal[i];
      disp[i * 3]     = origDisp[orig * 3];
      disp[i * 3 + 1] = origDisp[orig * 3 + 1];
      disp[i * 3 + 2] = origDisp[orig * 3 + 2];
    }
    return disp;
  }

  // Fallback: KD-tree nearest neighbour
  console.warn("[transfer] Using KD-tree fallback for displacement transfer");
  const indices = Array.from({ length: nOrig }, (_, i) => i);
  const tree = buildKDTree(indices, origNeutral);
  const disp = new Float32Array(nSimp * 3);
  for (let i = 0; i < nSimp; i++) {
    const best = { dist: Infinity, idx: 0 };
    kdNearest(tree, origNeutral,
      simpVerts[i * 3], simpVerts[i * 3 + 1], simpVerts[i * 3 + 2], best);
    disp[i * 3]     = origDisp[best.idx * 3];
    disp[i * 3 + 1] = origDisp[best.idx * 3 + 1];
    disp[i * 3 + 2] = origDisp[best.idx * 3 + 2];
  }
  return disp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Three.js mesh builder
// Build a Mesh with morphTarget set up from simplified verts + displacements.
// ─────────────────────────────────────────────────────────────────────────────

function buildMorphMesh(
  simpVerts: Float32Array,
  simpFaces: Uint32Array,
  dispVerts: Float32Array,   // simpVerts + displacement = smile positions
): THREE.Mesh {
  const geo = new THREE.BufferGeometry();

  geo.setAttribute("position", new THREE.BufferAttribute(simpVerts.slice(), 3));
  geo.setIndex(new THREE.BufferAttribute(simpFaces, 1));

  // Convert indexed → non-indexed so morph positions align 1:1
  const nonIndexed = geo.toNonIndexed();
  nonIndexed.computeVertexNormals();

  // Build smile positions for morph target.
  // We need to expand dispVerts through the same non-index expansion.
  // The non-indexed expansion follows the index order, so we replicate it:
  const indexArray = simpFaces;
  const nNonIdx = indexArray.length;  // = faces * 3

  const smilePos = new Float32Array(nNonIdx * 3);
  for (let i = 0; i < nNonIdx; i++) {
    const origIdx = indexArray[i];
    smilePos[i * 3]     = simpVerts[origIdx * 3]     + dispVerts[origIdx * 3];
    smilePos[i * 3 + 1] = simpVerts[origIdx * 3 + 1] + dispVerts[origIdx * 3 + 1];
    smilePos[i * 3 + 2] = simpVerts[origIdx * 3 + 2] + dispVerts[origIdx * 3 + 2];
  }

  const smileAttr = new THREE.BufferAttribute(smilePos, 3);
  nonIndexed.morphAttributes.position = [smileAttr];

  // Compute smile normals for morph target
  const smileGeo = new THREE.BufferGeometry();
  smileGeo.setAttribute("position", smileAttr);
  smileGeo.computeVertexNormals();
  nonIndexed.morphAttributes.normal = [
    smileGeo.attributes.normal as THREE.BufferAttribute,
  ];

  const mesh = new THREE.Mesh(nonIndexed, new THREE.MeshStandardMaterial({
    color: 0xdddddd,
    flatShading: false
  }));
  mesh.morphTargetInfluences = [0];
  return mesh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract raw vertex data from a loaded OBJ (before non-indexing)
// ─────────────────────────────────────────────────────────────────────────────

function extractIndexedVerts(obj: THREE.Group): {
  verts: Float64Array;
  faces: Int32Array;
} {
  const mesh = obj.children[0] as THREE.Mesh;
  // Apply the scale so positions match your existing code
  mesh.geometry.scale(0.89, 0.89, 0.89);

  const pos = mesh.geometry.attributes.position;
  const verts = new Float64Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    verts[i * 3]     = pos.getX(i);
    verts[i * 3 + 1] = pos.getY(i);
    verts[i * 3 + 2] = pos.getZ(i);
  }

  let faces: Int32Array;
  if (mesh.geometry.index) {
    faces = new Int32Array(mesh.geometry.index.array);
  } else {
    // Already non-indexed — generate trivial indices
    faces = Int32Array.from({ length: pos.count }, (_, i) => i);
  }

  return { verts, faces };
}

// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(msg: string) {
  const el = document.getElementById("status");
  // if (el) el.textContent = msg;
  if (el) el.innerHTML = msg;  // allow <br> for multi-line status
}

function setProgress(fraction: number) {
  const bar = document.getElementById("progress-bar") as HTMLElement | null;
  if (bar) bar.style.width = `${Math.round(fraction * 100)}%`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

const loader = new OBJLoader();
setStatus("Loading meshes…");

const [baseFaceObj, smileFaceObj] = await Promise.all([
  loader.loadAsync("ksHead/ksHeadNormal.obj"),
  loader.loadAsync("ksHead/ksHeadSmile.obj"),
]);

const { verts: origNeutral, faces: origFaces } = extractIndexedVerts(baseFaceObj);
const { verts: origSmile } = extractIndexedVerts(smileFaceObj);

console.log(`[Load] neutral: ${origNeutral.length / 3} verts, ${origFaces.length / 3} faces`);
console.log(`[Load] smile:   ${origSmile.length / 3} verts`);

setStatus("Ready — adjust ratio and click Apply.");

// Current mesh in the scene
let currentMesh: THREE.Mesh | null = null;

async function applySimplification(ratio: number) {
  setStatus("Simplifying…");
  setProgress(0);

  // Remove old mesh
  if (currentMesh) { scene.remove(currentMesh); currentMesh = null; }

  // Yield to browser so the UI updates before the heavy loop
  await new Promise(r => setTimeout(r, 16));

  const result = qemSimplify(origNeutral, origFaces, ratio, (f) => {
    setProgress(f);
    // Note: because QEM runs synchronously, progress only updates at yield points.
    // For very large meshes consider chunking with setTimeout between batches.
  });

  setStatus("Transferring smile displacement…");
  await new Promise(r => setTimeout(r, 4));

  const disp = transferDisplacement(
    origNeutral, origSmile,
    result.vertices, result.compactToOriginal,
  );

  setStatus("Building mesh…");
  await new Promise(r => setTimeout(r, 4));

  currentMesh = buildMorphMesh(result.vertices, result.faces, disp);
  scene.add(currentMesh);

  setProgress(1);
  setStatus(
    `Done — ${result.vertices.length / 3} verts, ` +
    `${result.faces.length / 3} faces (ratio ${ratio.toFixed(2)})`
  );
}

// ── UI wiring ────────────────────────────────────────────────────────────────

const ratioSlider   = document.getElementById("ratio-slider")   as HTMLInputElement;
const ratioDisplay  = document.getElementById("ratio-display")  as HTMLElement;
const morphSlider   = document.getElementById("morph-slider")   as HTMLInputElement;
const morphDisplay  = document.getElementById("morph-display")  as HTMLElement;
const applyBtn      = document.getElementById("apply-btn")      as HTMLButtonElement;

ratioSlider.addEventListener("input", () => {
  ratioDisplay.textContent = `${ratioSlider.value}%`;
});

morphSlider.addEventListener("input", () => {
  const t = parseFloat(morphSlider.value) / 100;
  morphDisplay.textContent = `${morphSlider.value}%`;
  if (currentMesh?.morphTargetInfluences) {
    currentMesh.morphTargetInfluences[0] = t;
  }
});

applyBtn.addEventListener("click", async () => {
  applyBtn.disabled = true;
  const ratio = parseInt(ratioSlider.value) / 100;
  await applySimplification(ratio);
  applyBtn.disabled = false;
});

// Load at full resolution on startup
await applySimplification(1.0);
if (ratioSlider) ratioSlider.value = "100";
if (ratioDisplay) ratioDisplay.textContent = "100%";

// ── Render loop ──────────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  directionalLight.position.copy(camera.position);
  renderer.render(scene, camera);
}
animate();