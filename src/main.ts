import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { qemSimplify } from "./qem";
import { extractIndexedVerts } from "./geometry-utils";

// ─────────────────────────────────────────────────────────────────────────────
// Scene setup
// ─────────────────────────────────────────────────────────────────────────────

// MARK: Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  3000,
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

// MARK: Build Morph Mesh
// ─────────────────────────────────────────────────────────────────────────────
// Build Three.js morph mesh
// ─────────────────────────────────────────────────────────────────────────────

const material = new THREE.MeshStandardMaterial({ color: 0xdddddd, flatShading: true });

function buildMorphMesh(
  simpVerts: Float32Array,
  simpFaces: Uint32Array,
  origNeutral: Float64Array,
  origSmile: Float64Array,
  compactToOriginal: Int32Array,
): THREE.Mesh {
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

    const dx = origSmile[orig * 3] - origNeutral[orig * 3];
    const dy = origSmile[orig * 3 + 1] - origNeutral[orig * 3 + 1];
    const dz = origSmile[orig * 3 + 2] - origNeutral[orig * 3 + 2];

    smilePos[i * 3] = simpVerts[i * 3] + dx;
    smilePos[i * 3 + 1] = simpVerts[i * 3 + 1] + dy;
    smilePos[i * 3 + 2] = simpVerts[i * 3 + 2] + dz;
  }

  const smileAttr = new THREE.BufferAttribute(smilePos, 3);
  geo.morphAttributes.position = [smileAttr];

  geo.computeVertexNormals();

  const mesh = new THREE.Mesh(
    geo,
    material,
  );
  mesh.morphTargetInfluences = [0];
  return mesh;
}

// ─────────────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(msg: string) {
  const el = document.getElementById("status");
  if (el) el.innerHTML = msg;
}
function setProgress(f: number) {
  const bar = document.getElementById("progress-bar") as HTMLElement | null;
  if (bar) bar.style.width = `${Math.round(f * 100)}%`;
}

// MARK: Main
// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

let origNeutral: Float64Array;
let origSmile: Float64Array;
let origFaces: Int32Array;

const loader = new OBJLoader();

async function loadModel(filename: string) {
  setStatus(`Loading ${filename}…`);
  setProgress(0);

  // Determine smile counterpart
  let smileFilename = filename;
  if (filename === "ksHeadNormal.obj") {
    smileFilename = "ksHeadSmile.obj";
  }

  const [baseFaceObj, smileFaceObj] = await Promise.all([
    loader.loadAsync(`ksHead/${filename}`),
    loader.loadAsync(`ksHead/${smileFilename}`),
  ]);

  const extracted = extractIndexedVerts(baseFaceObj, smileFaceObj);
  origNeutral = extracted.origNeutral;
  origSmile = extracted.origSmile;
  origFaces = extracted.origFaces;

  console.log(
    `[Load] ${filename}: ${origNeutral.length / 3} unique verts, ${origFaces.length / 3} faces`,
  );

  setStatus(`Loaded ${filename}. Ready to apply.`);
}

let meshes: THREE.Mesh[] = [];

async function applySimplification(ratio: number) {
  setStatus("Simplifying…");
  setProgress(0);
  meshes.forEach((m) => scene.remove(m));
  meshes = [];
  await new Promise((r) => setTimeout(r, 16));

  const start = performance.now();
  const result = qemSimplify(origNeutral, origFaces, ratio, (f) =>
    setProgress(f),
  );
  const end = performance.now();

  setStatus("Transferring smile…");
  await new Promise((r) => setTimeout(r, 4));

  setStatus("Building mesh…");
  await new Promise((r) => setTimeout(r, 4));
  const template = buildMorphMesh(
    result.vertices,
    result.faces,
    origNeutral,
    origSmile,
    result.compactToOriginal,
  );

  // Compute dynamic scale based on bounding box
  template.geometry.computeBoundingBox();
  const bbox = template.geometry.boundingBox!;
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);

  const targetSize = 20; // Desired max dimension
  const dynamicScale = targetSize / maxDim;

  // MARK: Clone Mesh
  const spacing = 30;
  for (let x = 0; x < 100; x++) {
    for (let z = 0; z < 10; z++) {
      const mesh = template.clone();
      mesh.scale.setScalar(dynamicScale);
      mesh.position.set((x - 4.5) * spacing, 0, (z - 4.5) * spacing);
      scene.add(mesh);
      meshes.push(mesh);
    }
  }

  setProgress(1);
  setStatus(`
    <div class="status-title qem-color">QEM</div>
    <div class="status-data">Time: ${(end - start).toFixed(0)}ms · Verts: ${result.vertices.length / 3} · Faces: ${result.faces.length / 3}</div>
  `);
}

const ratioSlider = document.getElementById("ratio-slider") as HTMLInputElement;
const ratioDisplay = document.getElementById("ratio-display") as HTMLElement;
const morphSlider = document.getElementById("morph-slider") as HTMLInputElement;
const morphDisplay = document.getElementById("morph-display") as HTMLElement;
const applyBtn = document.getElementById("apply-btn") as HTMLButtonElement;
const morphControls = document.getElementById("morph-controls") as HTMLElement;
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
const toggleShadingBtn = document.getElementById("toggle-shading-btn") as HTMLButtonElement;

toggleShadingBtn.addEventListener("click", () => {
  // Toggle the boolean
  material.flatShading = !material.flatShading;

  // CRITICAL: You must set this to true for the shader to re-compile 
  // with the new shading logic.
  material.needsUpdate = true;

  // Provide some feedback to the user
  const mode = material.flatShading ? "Flat" : "Smooth";
  console.log(`Shading mode changed to: ${mode}`);
});

ratioSlider.addEventListener("input", () => {
  ratioDisplay.textContent = `${ratioSlider.value}%`;
});

morphSlider.addEventListener("input", () => {
  morphDisplay.textContent = `${morphSlider.value}%`;
  meshes.forEach((m) => {
    if (m.morphTargetInfluences)
      m.morphTargetInfluences[0] = parseFloat(morphSlider.value) / 100;
  });
});

modelSelect.addEventListener("change", async () => {
  applyBtn.disabled = true;
  await loadModel(modelSelect.value);
  await applySimplification(parseInt(ratioSlider.value) / 100);
  applyBtn.disabled = false;
  checkMorphUI();
});

function checkMorphUI() {
  if (modelSelect.value === "ksHeadNormal.obj") {
    morphControls.style.display = "block";
  } else {
    morphControls.style.display = "none";
  }
}

applyBtn.addEventListener("click", async () => {
  applyBtn.disabled = true;
  morphDisplay.textContent = `0%`;
  morphSlider.value = "0";
  meshes.forEach((m) => {
    if (m.morphTargetInfluences) m.morphTargetInfluences[0] = 0;
  });
  await applySimplification(parseInt(ratioSlider.value) / 100);
  applyBtn.disabled = false;
});

// MARK: Initial load
await loadModel(modelSelect.value);
await applySimplification(1.0);
checkMorphUI();

ratioSlider.value = "100";
ratioDisplay.textContent = "100%";

const fpsDisplay = document.createElement("div");
fpsDisplay.style.position = "fixed";
fpsDisplay.style.top = "10px";
fpsDisplay.style.left = "10px";
fpsDisplay.style.color = "white";
fpsDisplay.style.backgroundColor = "rgba(20, 20, 24, 0.9)";
fpsDisplay.style.backdropFilter = "blur(12px)";
fpsDisplay.style.padding = "8px 16px";
fpsDisplay.style.borderRadius = "8px";
fpsDisplay.style.border = "1px solid rgba(255, 255, 255, 0.2)";
fpsDisplay.style.fontFamily = "'JetBrains Mono', monospace";
fpsDisplay.style.fontSize = "20px";
fpsDisplay.style.pointerEvents = "none";
fpsDisplay.style.zIndex = "1000";
fpsDisplay.style.boxShadow = "0 4px 16px rgba(0,0,0,0.5)";
document.body.appendChild(fpsDisplay);

let lastTime = performance.now();
let frames = 0;

function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  frames++;
  if (now > lastTime + 1000) {
    fpsDisplay.textContent = `FPS: ${Math.round((frames * 1000) / (now - lastTime))}`;
    lastTime = now;
    frames = 0;
  }

  // const t = (Math.sin(Date.now() * 0.001) + 1) / 2;
  // meshes.forEach((m) => {
  //   if (m.morphTargetInfluences) m.morphTargetInfluences[0] = t;
  // });
  controls.update();
  directionalLight.position.copy(camera.position);
  renderer.render(scene, camera);
}
animate();
