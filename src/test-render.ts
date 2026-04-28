import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createComparisonMeshes, simplifyGeometry } from "./test";
import { qemSimplify } from "./qem";

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

      // QEM
      const pos = faceGeo.attributes.position.array;
      const index = faceGeo.index!.array;
      const vertsIn = new Float64Array(pos);
      const facesIn = new Int32Array(index);

      const start2 = performance.now();
      const qemResult = qemSimplify(vertsIn, facesIn, ratio);
      const end2 = performance.now();

      const qemGeo = new THREE.BufferGeometry();
      qemGeo.setAttribute(
        "position",
        new THREE.BufferAttribute(qemResult.vertices, 3),
      );
      qemGeo.setIndex(new THREE.BufferAttribute(qemResult.faces, 1));
      qemGeo.computeVertexNormals();

      originalMesh.geometry.dispose();
      originalMesh.geometry = qemGeo;

      applyBtn.disabled = false;
      statusDisplay.textContent = `QEM: ${(end2 - start2).toFixed(0)}ms (${qemResult.vertices.length / 3} verts) | SimplifyModifier: ${(end1 - start1).toFixed(0)}ms (${newGeo.attributes.position.count} verts)`;
    }, 50);
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
