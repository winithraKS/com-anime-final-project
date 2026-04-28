import * as THREE from "three";
import { SimplifyModifier } from "three/addons/modifiers/SimplifyModifier.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils.js";

/**
 * Creates both original and simplified meshes for comparison.
 */
const modifier = new SimplifyModifier();

const loader = new OBJLoader();

/**
 * Simplifies a geometry by a percentage (0-1).
 */
export function simplifyGeometry(
  geometry: THREE.BufferGeometry,
  ratio: number,
) {
  const originalCount = geometry.attributes.position.count;
  const verticesToRemove = Math.floor(originalCount * ratio);

  if (verticesToRemove <= 0) return geometry.clone();

  return modifier.modify(geometry, verticesToRemove);
}

/**
 * Creates both original and simplified meshes for initial comparison.
 */
export async function createComparisonMeshes(filename: string = "ksHead/bunny30k.obj") {
  const group = await loader.loadAsync(filename);
  let faceGeo: THREE.BufferGeometry | null = null;

  group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      // OBJLoader returns non-indexed polygon soup. We MUST merge vertices
      // (weld them) so the SimplifyModifier can detect shared edges!
      faceGeo = BufferGeometryUtils.mergeVertices(child.geometry);
    }
  });

  if (!faceGeo) throw new Error("No mesh found in OBJ");

  // Type assertion needed because TS can't track assignments inside callbacks
  const validFaceGeo = faceGeo as THREE.BufferGeometry;

  const originalMaterial = new THREE.MeshStandardMaterial({
    color: 0xaa4444,
    wireframe: true,
  });
  const originalMesh = new THREE.Mesh(validFaceGeo, originalMaterial);

  // Initial state: No simplification to prevent browser freezing on load
  const simplifiedGeometry = validFaceGeo.clone();
  const simplifiedMaterial = new THREE.MeshStandardMaterial({
    color: 0x44aa88,
    wireframe: true,
  });
  const simplifiedMesh = new THREE.Mesh(simplifiedGeometry, simplifiedMaterial);

  return { originalMesh, simplifiedMesh, faceGeo: validFaceGeo };
}
