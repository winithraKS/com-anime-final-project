import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);
const canvas = document.querySelector<HTMLCanvasElement>("#bg");
if (!canvas) throw new Error('Canvas element "#bg" not found');
const renderer = new THREE.WebGLRenderer({
  canvas: canvas,
});

renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
camera.position.set(0, 10, -35);

renderer.render(scene, camera);
const controls = new OrbitControls(camera, renderer.domElement);

const directionalLight = new THREE.DirectionalLight();
scene.add(directionalLight);

const gridHelper = new THREE.GridHelper(200, 50);
scene.add(gridHelper);

// Load obj
const objLoader = new OBJLoader();
const baseFace = await objLoader.loadAsync("ksHead/ksHeadNormal.obj");
const smileFace = await objLoader.loadAsync("ksHead/ksHeadSmile.obj");

// Morph objects
const baseChild = baseFace.children[0] as THREE.Mesh;
const smileChild = smileFace.children[0] as THREE.Mesh;

baseChild.geometry.scale(0.89, 0.89, 0.89);
smileChild.geometry.scale(0.89, 0.89, 0.89);

baseChild.geometry = baseChild.geometry.toNonIndexed();
smileChild.geometry = smileChild.geometry.toNonIndexed();

const baseGeo = baseChild.geometry;
const smileGeo = smileChild.geometry;

console.log(
  "base:", baseGeo.attributes.position.count,
  "smile:", smileGeo.attributes.position.count
);

// make sure both are non-indexed (important!)
baseGeo.computeVertexNormals();
smileGeo.computeVertexNormals();

// 🔥 THIS is the key line
baseGeo.morphAttributes.position = [
  smileGeo.attributes.position
];

// optional but recommended (for lighting)
baseGeo.morphAttributes.normal = [
  smileGeo.attributes.normal
];

// init influence
baseChild.morphTargetInfluences = [1];

scene.add(baseChild);

let influence = 0;
document.addEventListener("keydown", (event) => {
  if (event.key === "w") {
    if (influence < 0.9) influence += 0.1;
  } else if (event.key === "s") {
    if (influence > 0.1) influence -= 0.1;
  }
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  directionalLight.position.copy(camera.position);

  if (baseChild.morphTargetInfluences) {
    baseChild.morphTargetInfluences[0] = influence;
  }

  renderer.render(scene, camera);
}

animate();
