import * as THREE from "three";

/**
 * Extract indexed verts from OBJ and weld them based on position.
 * Returns neutral positions, smile positions, and face indices.
 */
export function extractIndexedVerts(neutralObj: THREE.Group, smileObj: THREE.Group) {
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
  const seenIndex = new Uint8Array(nextIndex).fill(0);
  for (let i = 0; i < count; i++) {
    const newIdx = indices[i];
    if (seenIndex[newIdx] === 0) {
      sVertsFinal[newIdx * 3] = sPos.getX(i);
      sVertsFinal[newIdx * 3 + 1] = sPos.getY(i);
      sVertsFinal[newIdx * 3 + 2] = sPos.getZ(i);
      seenIndex[newIdx] = 1;
    }
  }

  return {
    origNeutral: nVertsFinal,
    origSmile: sVertsFinal,
    origFaces: new Int32Array(indices),
  };
}
