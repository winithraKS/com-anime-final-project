import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createComparisonMeshes, simplifyGeometry } from "./test";
import { qemSimplify } from "./qem";
import { extractIndexedVerts, buildKDTree, kdNearest, type KDNode } from "./geometry-utils";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

// 1. Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  3000,
);
camera.position.set(0, 10, 40);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 2. Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1);
pointLight.position.set(50, 50, 50);
scene.add(pointLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
scene.add(directionalLight);

// 3. Load meshes
let originalMesh: THREE.Mesh;
let simplifiedMesh: THREE.Mesh;
let faceGeo: THREE.BufferGeometry;

let origNeutral: Float64Array | null = null;
let origSmile: Float64Array | null = null;
let origFaces: Int32Array | null = null;
let currentCompactToOriginal: Int32Array | null = null;
let kdTree: KDNode | undefined = undefined;

const loader = new OBJLoader();

const modelSelect = document.getElementById(
  "model-select",
) as HTMLSelectElement;

async function loadModel(filename: string) {
  console.log(`Loading model: ${filename}...`);
  if (originalMesh) scene.remove(originalMesh);
  if (simplifiedMesh) scene.remove(simplifiedMesh);

  const result = await createComparisonMeshes(`ksHead/${filename}`);
  originalMesh = result.originalMesh;
  simplifiedMesh = result.simplifiedMesh;
  faceGeo = result.faceGeo;

  // Load smile if base.obj
  origNeutral = null;
  origSmile = null;
  origFaces = null;
  kdTree = undefined;
  if (filename === "base.obj") {
    const smileGroup = await loader.loadAsync("ksHead/smile.obj");
    const neutralGroup = await loader.loadAsync("ksHead/base.obj");
    const extracted = extractIndexedVerts(neutralGroup, smileGroup);
    origNeutral = extracted.origNeutral;
    origSmile = extracted.origSmile;
    origFaces = extracted.origFaces;

    // Build KD tree for fallback
    const nOrig = origNeutral.length / 3;
    const allIndices = Array.from({ length: nOrig }, (_, i) => i);
    kdTree = buildKDTree(allIndices, origNeutral);
  }

  // Dynamic scaling
  faceGeo.computeBoundingBox();
  const bbox = faceGeo.boundingBox!;
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);
  const targetSize = 20;
  const dynamicScale = targetSize / maxDim;

  originalMesh.scale.setScalar(dynamicScale);
  simplifiedMesh.scale.setScalar(dynamicScale);
  originalMesh.position.x = -10;
  simplifiedMesh.position.x = 10;

  scene.add(originalMesh);
  scene.add(simplifiedMesh);
  console.log("Model loaded and scaled.");
}

if (modelSelect) {
  modelSelect.addEventListener("change", async () => {
    await loadModel(modelSelect.value);
  });
}

// Initial load
await loadModel(modelSelect ? modelSelect.value : "bunny30k.obj");

// 5. Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);

// 6. Handle resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// 7. UI Controls
const toggleBtn = document.getElementById("toggle-wireframe");
if (toggleBtn) {
  toggleBtn.addEventListener("click", () => {
    const matOrig = originalMesh.material as THREE.MeshStandardMaterial;
    const matSimp = simplifiedMesh.material as THREE.MeshStandardMaterial;
    const isWireframe = matOrig.wireframe;
    matOrig.wireframe = !isWireframe;
    matSimp.wireframe = !isWireframe;
  });
}

const slider = document.getElementById(
  "simplification-slider",
) as HTMLInputElement;
const ratioValueDisplay = document.getElementById("ratio-value");
const applyBtn = document.getElementById("apply-btn") as HTMLButtonElement;
const statusDisplay = document.getElementById("status");

if (slider && ratioValueDisplay && applyBtn && statusDisplay) {
  slider.addEventListener("input", () => {
    ratioValueDisplay.textContent = slider.value;
  });

  applyBtn.addEventListener("click", () => {
    const ratio = parseInt(slider.value) / 100;
    applyBtn.disabled = true;
    statusDisplay.textContent = "Simplifying...";

    // Use timeout to allow UI update
    setTimeout(() => {
      // SimplifyModifier
      const start1 = performance.now();
      const newGeo = simplifyGeometry(faceGeo, 1 - ratio);
      const end1 = performance.now();

      simplifiedMesh.geometry.dispose();
      simplifiedMesh.geometry = newGeo;

      // KD-TREE MORPH FOR SIMPLIFYMODIFIER
      if (origNeutral && origSmile && kdTree) {
        const morphPos = new Float32Array(newGeo.attributes.position.count * 3);
        const origDisp = new Float32Array(origNeutral.length);
        for (let i = 0; i < origNeutral.length; i++) {
          origDisp[i] = origSmile[i] - origNeutral[i];
        }

        for (let i = 0; i < morphPos.length / 3; i++) {
          const qx = newGeo.attributes.position.getX(i);
          const qy = newGeo.attributes.position.getY(i);
          const qz = newGeo.attributes.position.getZ(i);

          const best = { dist: Infinity, idx: 0 };
          kdNearest(kdTree, origNeutral, qx, qy, qz, best);

          const origIdx = best.idx;
          morphPos[i * 3] = qx + origDisp[origIdx * 3];
          morphPos[i * 3 + 1] = qy + origDisp[origIdx * 3 + 1];
          morphPos[i * 3 + 2] = qz + origDisp[origIdx * 3 + 2];
        }
        newGeo.morphAttributes.position = [new THREE.BufferAttribute(morphPos, 3)];
        simplifiedMesh.morphTargetInfluences = [0];
      }

      // QEM
      let vertsIn: Float64Array;
      let facesIn: Int32Array;

      if (origNeutral && origFaces) {
        vertsIn = origNeutral;
        facesIn = origFaces;
      } else {
        const pos = faceGeo.attributes.position.array;
        const index = faceGeo.index!.array;
        vertsIn = new Float64Array(pos);
        facesIn = new Int32Array(index);
      }

      const start2 = performance.now();
      const qemResult = qemSimplify(vertsIn, facesIn, ratio);
      const end2 = performance.now();
      currentCompactToOriginal = qemResult.compactToOriginal;

      const qemGeo = new THREE.BufferGeometry();
      qemGeo.setAttribute(
        "position",
        new THREE.BufferAttribute(qemResult.vertices, 3),
      );
      qemGeo.setIndex(new THREE.BufferAttribute(qemResult.faces, 1));

      // CORRECT MORPH FOR QEM
      if (origNeutral && origSmile && currentCompactToOriginal) {
        const morphPos = new Float32Array(qemResult.vertices.length);
        for (let i = 0; i < qemResult.vertices.length / 3; i++) {
          const orig = currentCompactToOriginal[i];
          const dx = origSmile[orig * 3] - origNeutral[orig * 3];
          const dy = origSmile[orig * 3 + 1] - origNeutral[orig * 3 + 1];
          const dz = origSmile[orig * 3 + 2] - origNeutral[orig * 3 + 2];
          morphPos[i * 3] = qemResult.vertices[i * 3] + dx;
          morphPos[i * 3 + 1] = qemResult.vertices[i * 3 + 1] + dy;
          morphPos[i * 3 + 2] = qemResult.vertices[i * 3 + 2] + dz;
        }
        qemGeo.morphAttributes.position = [new THREE.BufferAttribute(morphPos, 3)];
        originalMesh.morphTargetInfluences = [0];
      }

      qemGeo.computeVertexNormals();
      originalMesh.geometry.dispose();
      originalMesh.geometry = qemGeo;

      applyBtn.disabled = false;
      statusDisplay.textContent = `QEM: ${(end2 - start2).toFixed(0)}ms (${qemResult.vertices.length / 3} verts) | SimplifyModifier: ${(end1 - start1).toFixed(0)}ms (${newGeo.attributes.position.count} verts)`;
    }, 50);
  });
}

const morphSlider = document.getElementById("morph-slider") as HTMLInputElement;
const morphValueDisplay = document.getElementById("morph-value");

if (morphSlider && morphValueDisplay) {
  morphSlider.addEventListener("input", () => {
    const val = parseInt(morphSlider.value) / 100;
    morphValueDisplay.textContent = `${morphSlider.value}%`;
    if (originalMesh.morphTargetInfluences) {
      originalMesh.morphTargetInfluences[0] = val;
    }
    if (simplifiedMesh.morphTargetInfluences) {
      simplifiedMesh.morphTargetInfluences[0] = val;
    }
  });
}

// 8. Animation loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

console.log("Test render started");
