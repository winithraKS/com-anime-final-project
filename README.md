https://github.com/winithraKS/com-anime-final-project
<img width="555" height="538" alt="demo v1" src="https://github.com/user-attachments/assets/bb2fda97-b0cc-4fe9-90c4-e156c63eebea" />

# Real-time Facial Mesh Simplification with Morph Target Preservation

---

## Problem
High-resolution 3D scans and photogrammetry produce models with millions of polygons, which are unusable in interactive web applications due to **GPU memory constraints** and **rendering overhead**. Standard tools like the Three.js `SimplifyModifier()` fail to maintain a consistent **60 FPS** because their $O(N^2)$ scaling causes browsers to freeze when processing dense meshes. Furthermore, these methods often "melt" critical facial features and break animation capabilities, making them unsuitable for high-quality character rendering.

---

## Approach

### **Baseline: SimplifyModifier()**
*   **Metric**: Uses simple **Edge Length** and distance-based clustering to determine importance.
*   **Data Structure**: Relies on **flat arrays** and linear search logic to track vertex merges.
*   **Vertex Placement**: Employs "folding" (midpoint or endpoint selection) rather than mathematical optimization.

### **Proposed: Optimized QEM (Quadric Error Metrics)**
*   **Surface Mapping**: Store matrices representing squared distances to all neighboring face planes.
*   **Error Assessment**: Use a heap to find the least significant edge and solve a linear equation for new vertex coordinates.
*   **Algorithmic Efficiency**: Maintain $O(N \log N)$ complexity by using a **Min-Heap** for logarithmic updates and **Union-Find** for instant vertex tracking.
*   **Boundary Constraints**: Validate new vertices against safe boundaries to prevent needle-like geometric glitches.
*   **Animation Persistence**: Retain links to the original model for mapping facial morphs onto geometry.



---

## Results

Benchmarks conducted on **Apple M2** hardware using the high-density `ksHeadNormal.obj` dataset show significant improvements:

| Metric | Result |
| :--- | :--- |
| **Speedup (High-Poly)** | **93% – 95% faster** than the SimplifyModifier baseline. |
| **Stability Improvement** | **100% Manifold Integrity**; successfully processed meshes where the baseline crashed. |
| **Complexity Scaling** | Reduced from **$O(N^2)$** to **$O(N \log N)$** efficiency. |
| **Animation Quality** | Successfully preserved facial expressions (smile) across all LOD levels. |

---

## How to Run

Follow these steps to initialize the simplification environment and run the benchmarks:

### **1. Environment Setup**
Ensure you have **Node.js** installed. Clone the repository and install the necessary dependencies:
```bash
npm install
```

### **2. Running the Local Server**
Launch the development server:
```bash
npm run dev
```
Navigate to `http://localhost:5173` in your browser.

### **3. Executing the Simplification**
*   **Set Ratio**: Adjust the simplification slider (e.g., set to 0.1 for 90% reduction).
*   **Apply QEM**: Click **"Apply"** to trigger the simplify algorithm.
*   **Benchmark**: View the `Benchmark Result` page to see the execution time comparison between the Proposed QEM and the Baseline.

### **4. Animation Test**
* **Smile Slider**: Adjust the slider to morph between smile face and neutral face. (only available for `frame_base.obj` and `ksHeadNormal.obj`)