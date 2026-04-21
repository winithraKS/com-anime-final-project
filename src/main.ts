import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils.js";

// ─────────────────────────────────────────────────────────────────────────────
// Scene setup
// ─────────────────────────────────────────────────────────────────────────────

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
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

interface SimplifyResult {
  vertices: Float32Array;
  faces: Uint32Array;
  compactToOriginal: Int32Array;
}

// ─────────────────────────────────────────────────────────────────────────────
// QEM helpers
// ─────────────────────────────────────────────────────────────────────────────

function computeQuadrics(verts: Float64Array, faces: Int32Array, nVerts: number): Float64Array {
  const Q = new Float64Array(nVerts * 16);
  const nFaces = faces.length / 3;
  for (let fi = 0; fi < nFaces; fi++) {
    const i0 = faces[fi*3], i1 = faces[fi*3+1], i2 = faces[fi*3+2];
    const ax = verts[i0*3], ay = verts[i0*3+1], az = verts[i0*3+2];
    const bx = verts[i1*3], by = verts[i1*3+1], bz = verts[i1*3+2];
    const cx = verts[i2*3], cy = verts[i2*3+1], cz = verts[i2*3+2];
    let nx = (by-ay)*(cz-az) - (bz-az)*(cy-ay);
    let ny = (bz-az)*(cx-ax) - (bx-ax)*(cz-az);
    let nz = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    const len = Math.sqrt(nx*nx + ny*ny + nz*nz);
    if (len < 1e-12) continue;
    nx /= len; ny /= len; nz /= len;
    const d = -(nx*ax + ny*ay + nz*az);
    const p = [nx, ny, nz, d];
    for (const vi of [i0, i1, i2]) {
      const base = vi * 16;
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
          Q[base + r*4 + c] += p[r] * p[c];
    }
  }
  return Q;
}

function evalCost(Qc: Float64Array, x: number, y: number, z: number): number {
  const v = [x, y, z, 1.0];
  let cost = 0;
  for (let r = 0; r < 4; r++) {
    let row = 0;
    for (let c = 0; c < 4; c++) row += Qc[r*4+c] * v[c];
    cost += row * v[r];
  }
  return cost;
}

function solve3(A: number[], b: number[], out: number[]): boolean {
  const M = [
    [A[0], A[1], A[2], b[0]],
    [A[3], A[4], A[5], b[1]],
    [A[6], A[7], A[8], b[2]],
  ];
  for (let col = 0; col < 3; col++) {
    let maxVal = Math.abs(M[col][col]), maxRow = col;
    for (let row = col+1; row < 3; row++) {
      if (Math.abs(M[row][col]) > maxVal) { maxVal = Math.abs(M[row][col]); maxRow = row; }
    }
    if (maxVal < 1e-10) return false;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    const inv = 1 / M[col][col];
    for (let row = col+1; row < 3; row++) {
      const f = M[row][col] * inv;
      for (let k = col; k <= 3; k++) M[row][k] -= f * M[col][k];
    }
  }
  for (let row = 2; row >= 0; row--) {
    let s = M[row][3];
    for (let k = row+1; k < 3; k++) s -= M[row][k] * out[k];
    out[row] = s / M[row][row];
  }
  return true;
}

/**
 * Find the best collapse position for edge (vi, vj).
 *
 * Strategy: evaluate up to 4 candidates and pick the one with lowest cost.
 *   1. Optimal point from solving the linear system  (if solvable AND close)
 *   2. vi position
 *   3. vj position
 *   4. midpoint
 *
 * The key safety check: reject the optimal point if it lies more than
 * (edgeLength * MAX_STRETCH) away from vi. On flat/smooth regions the
 * quadric is nearly singular and the "optimal" point can fly off to
 * infinity, causing the exploded mesh artifact.
 */
function optimalVertex(
  Qc: Float64Array,
  vix: number, viy: number, viz: number,
  vjx: number, vjy: number, vjz: number,
): Vec3 {
  const midpoint: Vec3 = [(vix + vjx) * 0.5, (viy + vjy) * 0.5, (viz + vjz) * 0.5];
  const candidates: Vec3[] = [[vix, viy, viz], [vjx, vjy, vjz], midpoint];

  const A = [Qc[0], Qc[1], Qc[2], Qc[4], Qc[5], Qc[6], Qc[8], Qc[9], Qc[10]];
  const b = [-Qc[3], -Qc[7], -Qc[11]];
  const sol = [0.0, 0.0, 0.0];

  if (solve3(A, b, sol)) {
    // 1. Calculate Bounding Box of the edge
    const minX = Math.min(vix, vjx), maxX = Math.max(vix, vjx);
    const minY = Math.min(viy, vjy), maxY = Math.max(viy, vjy);
    const minZ = Math.min(viz, vjz), maxZ = Math.max(viz, vjz);

    // 2. Add a small "padding" to the box (e.g., 20% of edge length)
    const edgeLen = Math.sqrt((vjx - vix) ** 2 + (vjy - viy) ** 2 + (vjz - viz) ** 2);
    const pad = edgeLen * 0.2;

    // 3. ONLY accept optimal point if it's inside this padded box
    if (sol[0] >= minX - pad && sol[0] <= maxX + pad &&
        sol[1] >= minY - pad && sol[1] <= maxY + pad &&
        sol[2] >= minZ - pad && sol[2] <= maxZ + pad) {
      candidates.push(sol as Vec3);
    }
  }

  let best = candidates[0];
  let bestCost = Infinity;
  for (const c of candidates) {
    const cost = evalCost(Qc, c[0], c[1], c[2]);
    if (cost < bestCost) {
      bestCost = cost;
      best = c;
    }
  }
  return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Min-heap
// ─────────────────────────────────────────────────────────────────────────────

class MinHeap<T> {
  private data: { key: number; val: T }[] = [];
  push(key: number, val: T) { this.data.push({ key, val }); this._up(this.data.length - 1); }
  pop() {
    if (!this.data.length) return undefined;
    const top = this.data[0];
    const last = this.data.pop()!;
    if (this.data.length) { this.data[0] = last; this._down(0); }
    return top;
  }
  get size() { return this.data.length; }
  private _up(i: number) {
    while (i > 0) {
      const p = (i-1)>>1;
      if (this.data[p].key <= this.data[i].key) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]]; i = p;
    }
  }
  private _down(i: number) {
    const n = this.data.length;
    while (true) {
      let s = i; const l = 2*i+1, r = 2*i+2;
      if (l < n && this.data[l].key < this.data[s].key) s = l;
      if (r < n && this.data[r].key < this.data[s].key) s = r;
      if (s === i) break;
      [this.data[s], this.data[i]] = [this.data[i], this.data[s]]; i = s;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// QEM simplification
// ─────────────────────────────────────────────────────────────────────────────

export function qemSimplify(
  vertsIn: Float64Array,
  facesIn: Int32Array,
  ratio: number,
  onProgress?: (f: number) => void,
): SimplifyResult {
  
  const nVertsOrig = vertsIn.length / 3;
  const nFacesOrig = facesIn.length / 3;
  const targetFaces = Math.max(1, Math.floor(nFacesOrig * ratio));

  const pos = new Float64Array(vertsIn);
  const Q   = computeQuadrics(pos, facesIn, nVertsOrig);

  // Union-Find with full path compression
  const parent = Int32Array.from({ length: nVertsOrig }, (_, i) => i);
  function find(x: number): number {
    let root = x;
    while (parent[root] !== root) root = parent[root];
    while (parent[x] !== root) { const nx = parent[x]; parent[x] = root; x = nx; }
    return root;
  }
  function union(keep: number, remove: number, nx: number, ny: number, nz: number) {
    parent[remove] = keep;
    pos[keep*3] = nx; pos[keep*3+1] = ny; pos[keep*3+2] = nz;
    for (let k = 0; k < 16; k++) Q[keep*16+k] += Q[remove*16+k];
  }

  // Face storage — corners kept as current roots
  const faces     = new Int32Array(facesIn);
  const faceAlive = new Uint8Array(nFacesOrig).fill(1);
  let nFaces = nFacesOrig;

  // Vertex → face adjacency
  const vertFaces: Set<number>[] = Array.from({ length: nVertsOrig }, () => new Set());
  for (let fi = 0; fi < nFacesOrig; fi++) {
    vertFaces[faces[fi*3]].add(fi);
    vertFaces[faces[fi*3+1]].add(fi);
    vertFaces[faces[fi*3+2]].add(fi);
  }

  // Replace your edgeKey with this (Max 2M vertices)
  const edgeKey = (u: number, v: number) => {
    return u < v ? (u * 2000003 + v) : (v * 2000003 + u);
  };

  const edgeSeen = new Set<number>();
  const heap = new MinHeap<[number, number, Vec3]>();

  function pushEdge(vi: number, vj: number) {
    // Always ensure we are using current ROOTS
    vi = find(vi); 
    vj = find(vj);
    if (vi === vj) return;

    const key = edgeKey(vi, vj);
    if (edgeSeen.has(key)) return;
    edgeSeen.add(key);

    const Qc = new Float64Array(16);
    for (let k = 0; k < 16; k++) Qc[k] = Q[vi * 16 + k] + Q[vj * 16 + k];
    
    const opt = optimalVertex(Qc, pos[vi*3], pos[vi*3+1], pos[vi*3+2], pos[vj*3], pos[vj*3+1], pos[vj*3+2]);
    heap.push(evalCost(Qc, opt[0], opt[1], opt[2]), [vi, vj, opt]);
  }

  for (let fi = 0; fi < nFacesOrig; fi++) {
    const a = faces[fi*3], b = faces[fi*3+1], c = faces[fi*3+2];
    pushEdge(a, b); pushEdge(b, c); pushEdge(a, c);
  }

  const totalToRemove = nFacesOrig - targetFaces;
  let removed = 0;

  while (nFaces > targetFaces && heap.size > 0) {
    const item = heap.pop()!;
    const [vi_orig, vj_orig, vopt] = item.val;

    let vi = find(vi_orig); 
    let vj = find(vj_orig);
    
    // 1. STALE CHECK: Discard if roots have already merged
    if (vi === vj) continue;
    if (vi !== vi_orig || vj !== vj_orig) continue; 

    // 2. UNION & POSITION UPDATE
    union(vi, vj, vopt[0], vopt[1], vopt[2]);

    // 3. RECOMPUTE QUADRIC FROM LIVE FACES (Stops the "Spikes")
    // We clear it, then sum only active triangles to avoid double-counting.
    for (let k = 0; k < 16; k++) Q[vi * 16 + k] = 0;
    const newNeighbors = new Set<number>();

    // Combine vj's faces into vi
    for (const fi of vertFaces[vj]) {
      if (!faceAlive[fi]) continue;
      
      // Update all corners to current roots
      const c0 = find(faces[fi*3]);
      const c1 = find(faces[fi*3+1]);
      const c2 = find(faces[fi*3+2]);
      faces[fi*3] = c0; faces[fi*3+1] = c1; faces[fi*3+2] = c2;

      if (c0 === c1 || c1 === c2 || c0 === c2) {
        faceAlive[fi] = 0;
        nFaces--;
        removed++;
        // Scrub face from neighbors
        for (const c of [c0, c1, c2]) vertFaces[c].delete(fi);
      } else {
        vertFaces[vi].add(fi);
        newNeighbors.add(c0); newNeighbors.add(c1); newNeighbors.add(c2);
      }
    }
    vertFaces[vj].clear();

    // 4. SECOND PASS: Finalize vi's quadrics and neighbors
    for (const fi of vertFaces[vi]) {
      if (!faceAlive[fi]) continue;
      
      // Re-sum the quadric for THIS specific live face
      // (This logic mirrors your computeQuadrics function)
      const i0 = faces[fi*3], i1 = faces[fi*3+1], i2 = faces[fi*3+2];
      const ax = pos[i0*3], ay = pos[i0*3+1], az = pos[i0*3+2];
      const bx = pos[i1*3], by = pos[i1*3+1], bz = pos[i1*3+2];
      const cx = pos[i2*3], cy = pos[i2*3+1], cz = pos[i2*3+2];
      let nx = (by-ay)*(cz-az) - (bz-az)*(cy-ay);
      let ny = (bz-az)*(cx-ax) - (bx-ax)*(cz-az);
      let nz = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
      const len = Math.sqrt(nx*nx + ny*ny + nz*nz);
      if (len > 1e-12) {
        nx /= len; ny /= len; nz /= len;
        const d = -(nx*ax + ny*ay + nz*az);
        const p = [nx, ny, nz, d];
        for (let r = 0; r < 4; r++)
          for (let c = 0; c < 4; c++)
            Q[vi * 16 + r*4 + c] += p[r] * p[c];
      }
      newNeighbors.add(i0); newNeighbors.add(i1); newNeighbors.add(i2);
    }

    // 5. RE-PUSH EDGES (Ensures app doesn't hang)
    newNeighbors.delete(vi);
    for (const nb of newNeighbors) {
      const targetNb = find(nb);
      if (targetNb !== vi) {
        // Clear the key so pushEdge can update the cost in the heap
        edgeSeen.delete(edgeKey(vi, targetNb)); 
        pushEdge(vi, targetNb);
      }
    }

    if (onProgress && removed % 500 === 0)
      onProgress(Math.min(removed / totalToRemove, 1));
  }

  // Compact
  const validFacesList: number[] = [];
  for (let fi = 0; fi < nFacesOrig; fi++) {
    if (!faceAlive[fi]) continue;
    const a = find(faces[fi*3]), b = find(faces[fi*3+1]), c = find(faces[fi*3+2]);
    if (a !== b && b !== c && a !== c) validFacesList.push(a, b, c);
  }

  const usedRoots = [...new Set(validFacesList)].sort((a, b) => a - b);
  const rootToCompact = new Map<number, number>();
  usedRoots.forEach((r, i) => rootToCompact.set(r, i));

  const nCompact = usedRoots.length;
  const compactVerts = new Float32Array(nCompact * 3);
  const compactToOriginal = new Int32Array(nCompact);
  for (let i = 0; i < nCompact; i++) {
    const r = usedRoots[i];
    compactVerts[i*3] = pos[r*3]; compactVerts[i*3+1] = pos[r*3+1]; compactVerts[i*3+2] = pos[r*3+2];
    compactToOriginal[i] = r;
  }

  const compactFaces = new Uint32Array(validFacesList.length);
  for (let i = 0; i < validFacesList.length; i++)
    compactFaces[i] = rootToCompact.get(validFacesList[i])!;

  onProgress?.(1);
  return { vertices: compactVerts, faces: compactFaces, compactToOriginal };
}

// // ─────────────────────────────────────────────────────────────────────────────
// // KD-tree
// // ─────────────────────────────────────────────────────────────────────────────

// interface KDNode { idx: number; axis: number; left?: KDNode; right?: KDNode; }

// function buildKDTree(indices: number[], verts: Float64Array, depth = 0): KDNode | undefined {
//   if (!indices.length) return undefined;
//   const axis = depth % 3;
//   indices.sort((a, b) => verts[a*3+axis] - verts[b*3+axis]);
//   const mid = indices.length >> 1;
//   return {
//     idx: indices[mid], axis,
//     left:  buildKDTree(indices.slice(0, mid),   verts, depth+1),
//     right: buildKDTree(indices.slice(mid+1),     verts, depth+1),
//   };
// }

// function kdNearest(node: KDNode | undefined, verts: Float64Array, qx: number, qy: number, qz: number, best: { dist: number; idx: number }) {
//   if (!node) return;
//   const px = verts[node.idx*3], py = verts[node.idx*3+1], pz = verts[node.idx*3+2];
//   const d = (qx-px)**2 + (qy-py)**2 + (qz-pz)**2;
//   if (d < best.dist) { best.dist = d; best.idx = node.idx; }
//   const diff = [qx,qy,qz][node.axis] - [px,py,pz][node.axis];
//   const near = diff < 0 ? node.left : node.right;
//   const far  = diff < 0 ? node.right : node.left;
//   kdNearest(near, verts, qx, qy, qz, best);
//   if (diff*diff < best.dist) kdNearest(far, verts, qx, qy, qz, best);
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Displacement transfer
// // ─────────────────────────────────────────────────────────────────────────────

// function transferDisplacement(origNeutral: Float64Array, origSmile: Float64Array, simpVerts: Float32Array, compactToOriginal: Int32Array): Float32Array {
//   const nOrig = origNeutral.length / 3;
//   const nSimp = simpVerts.length / 3;
//   const origDisp = new Float32Array(nOrig * 3);
//   for (let i = 0; i < nOrig*3; i++) origDisp[i] = origSmile[i] - origNeutral[i];

//   if (compactToOriginal.every(idx => idx < nOrig)) {
//     const disp = new Float32Array(nSimp * 3);
//     for (let i = 0; i < nSimp; i++) {
//       const o = compactToOriginal[i];
//       disp[i*3] = origDisp[o*3]; disp[i*3+1] = origDisp[o*3+1]; disp[i*3+2] = origDisp[o*3+2];
//     }
//     return disp;
//   }

//   console.warn("[transfer] KD-tree fallback");
//   const tree = buildKDTree(Array.from({ length: nOrig }, (_, i) => i), origNeutral);
//   const disp = new Float32Array(nSimp * 3);
//   for (let i = 0; i < nSimp; i++) {
//     const best = { dist: Infinity, idx: 0 };
//     kdNearest(tree, origNeutral, simpVerts[i*3], simpVerts[i*3+1], simpVerts[i*3+2], best);
//     disp[i*3] = origDisp[best.idx*3]; disp[i*3+1] = origDisp[best.idx*3+1]; disp[i*3+2] = origDisp[best.idx*3+2];
//   }
//   return disp;
// }

// ─────────────────────────────────────────────────────────────────────────────
// Build Three.js morph mesh
// ─────────────────────────────────────────────────────────────────────────────

function buildMorphMesh(simpVerts: Float32Array, simpFaces: Uint32Array, origNeutral: Float64Array, origSmile: Float64Array, compactToOriginal: Int32Array): THREE.Mesh {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(simpVerts, 3));
  geo.setIndex(new THREE.BufferAttribute(simpFaces, 1));

  // build morph target (same indexing!)
  const smilePos = new Float32Array(simpVerts.length);

  for (let i = 0; i < simpVerts.length / 3; i++) {
    // smilePos[i*3]     = simpVerts[i*3]     + dispVerts[i*3];
    // smilePos[i*3 + 1] = simpVerts[i*3 + 1] + dispVerts[i*3 + 1];
    // smilePos[i*3 + 2] = simpVerts[i*3 + 2] + dispVerts[i*3 + 2];
    const orig = compactToOriginal[i];

    const dx = origSmile[orig*3]     - origNeutral[orig*3];
    const dy = origSmile[orig*3 + 1] - origNeutral[orig*3 + 1];
    const dz = origSmile[orig*3 + 2] - origNeutral[orig*3 + 2];

    smilePos[i*3]     = simpVerts[i*3]     + dx;
    smilePos[i*3 + 1] = simpVerts[i*3 + 1] + dy;
    smilePos[i*3 + 2] = simpVerts[i*3 + 2] + dz;
  }

  const smileAttr = new THREE.BufferAttribute(smilePos, 3);
  geo.morphAttributes.position = [smileAttr];

  geo.computeVertexNormals();

  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xdddddd, flatShading: true }));
  mesh.morphTargetInfluences = [0];
  return mesh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract indexed verts from OBJ
// ─────────────────────────────────────────────────────────────────────────────

function extractIndexedVerts(neutralObj: THREE.Group, smileObj: THREE.Group) {
  const nMesh = neutralObj.children[0] as THREE.Mesh;
  const sMesh = smileObj.children[0] as THREE.Mesh;

  const nPos = nMesh.geometry.attributes.position;
  const sPos = sMesh.geometry.attributes.position;
  const count = nPos.count;

  const uniqueVerts: number[] = [];
  const indices = new Int32Array(count);
  const hashTable = new Map<string, number>();
  
  // Precision for welding (snaps vertices within 0.0001 units)
  const precision = 10000; 

  let nextIndex = 0;
  for (let i = 0; i < count; i++) {
    const x = nPos.getX(i);
    const y = nPos.getY(i);
    const z = nPos.getZ(i);
    
    // Create a string key based on rounded coordinates
    const key = `${Math.round(x * precision)}_${Math.round(y * precision)}_${Math.round(z * precision)}`;

    if (hashTable.has(key)) {
      indices[i] = hashTable.get(key)!;
    } else {
      hashTable.set(key, nextIndex);
      indices[i] = nextIndex;
      uniqueVerts.push(x, y, z);
      nextIndex++;
    }
  }

  const nVertsFinal = new Float64Array(uniqueVerts);
  const sVertsFinal = new Float64Array(nextIndex * 3);

  // Map smile vertices to the NEW unique indices
  // We use the first occurrence of a "welded" vertex to represent the smile position
  const seenIndex = new Uint8Array(nextIndex).fill(0);
  for (let i = 0; i < count; i++) {
    const newIdx = indices[i];
    if (seenIndex[newIdx] === 0) {
      sVertsFinal[newIdx * 3]     = sPos.getX(i);
      sVertsFinal[newIdx * 3 + 1] = sPos.getY(i);
      sVertsFinal[newIdx * 3 + 2] = sPos.getZ(i);
      seenIndex[newIdx] = 1;
    }
  }

  return { 
    origNeutral: nVertsFinal, 
    origSmile: sVertsFinal, 
    origFaces: new Int32Array(indices) 
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(msg: string) { const el = document.getElementById("status"); if (el) el.innerHTML = msg; }
function setProgress(f: number) { const bar = document.getElementById("progress-bar") as HTMLElement | null; if (bar) bar.style.width = `${Math.round(f*100)}%`; }

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

const loader = new OBJLoader();
setStatus("Loading meshes…");

const [baseFaceObj, smileFaceObj] = await Promise.all([
  loader.loadAsync("ksHead/ksHeadNormal.obj"),
  loader.loadAsync("ksHead/ksHeadSmile.obj"),
]);
const { origNeutral, origSmile, origFaces } = extractIndexedVerts(baseFaceObj, smileFaceObj);
console.log(`[Load] neutral: ${origNeutral.length} verts, ${origFaces.length} faces`);
console.log(`[Load] smile: ${origSmile.length} verts`)

setStatus("Ready — adjust ratio and click Apply.");
let currentMesh: THREE.Mesh | null = null;

async function applySimplification(ratio: number) {
  setStatus("Simplifying…");
  setProgress(0);
  if (currentMesh) { scene.remove(currentMesh); currentMesh = null; }
  await new Promise(r => setTimeout(r, 16));

  const result = qemSimplify(origNeutral, origFaces, ratio, f => setProgress(f));

  setStatus("Transferring smile…");
  await new Promise(r => setTimeout(r, 4));
  // const disp = transferDisplacement(origNeutral, origSmile, result.vertices, result.compactToOriginal);

  setStatus("Building mesh…");
  await new Promise(r => setTimeout(r, 4));
  currentMesh = buildMorphMesh(result.vertices, result.faces, origNeutral, origSmile, result.compactToOriginal);
  scene.add(currentMesh);

  setProgress(1);
  setStatus(`${result.vertices.length/3} verts · ${result.faces.length/3} faces · ratio ${ratio.toFixed(2)}`);
}

const ratioSlider  = document.getElementById("ratio-slider")  as HTMLInputElement;
const ratioDisplay = document.getElementById("ratio-display") as HTMLElement;
const morphSlider  = document.getElementById("morph-slider")  as HTMLInputElement;
const morphDisplay = document.getElementById("morph-display") as HTMLElement;
const applyBtn     = document.getElementById("apply-btn")     as HTMLButtonElement;

ratioSlider.addEventListener("input", () => { ratioDisplay.textContent = `${ratioSlider.value}%`; });
morphSlider.addEventListener("input", () => {
  morphDisplay.textContent = `${morphSlider.value}%`;
  if (currentMesh?.morphTargetInfluences) currentMesh.morphTargetInfluences[0] = parseFloat(morphSlider.value) / 100;
});
applyBtn.addEventListener("click", async () => {
  applyBtn.disabled = true;
  morphDisplay.textContent = `0%`;
  morphSlider.value = '0';
  if (currentMesh?.morphTargetInfluences) currentMesh.morphTargetInfluences[0] = 0;
  await applySimplification(parseInt(ratioSlider.value) / 100);
  applyBtn.disabled = false;
});

await applySimplification(1.0);
ratioSlider.value = "100";
ratioDisplay.textContent = "100%";

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  directionalLight.position.copy(camera.position);
  renderer.render(scene, camera);
}
animate();