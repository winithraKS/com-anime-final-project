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

export interface KDNode {
  idx: number;
  axis: number;
  left?: KDNode;
  right?: KDNode;
}

export function buildKDTree(indices: number[], verts: Float64Array, depth = 0): KDNode | undefined {
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

export function kdNearest(
  node: KDNode | undefined,
  verts: Float64Array,
  qx: number,
  qy: number,
  qz: number,
  best: { dist: number; idx: number }
) {
  if (!node) return;
  const px = verts[node.idx * 3],
    py = verts[node.idx * 3 + 1],
    pz = verts[node.idx * 3 + 2];
  const d = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2;
  if (d < best.dist) {
    best.dist = d;
    best.idx = node.idx;
  }
  const diff = [qx, qy, qz][node.axis] - [px, py, pz][node.axis];
  const near = diff < 0 ? node.left : node.right;
  const far = diff < 0 ? node.right : node.left;
  kdNearest(near, verts, qx, qy, qz, best);
  if (diff * diff < best.dist) kdNearest(far, verts, qx, qy, qz, best);
}
