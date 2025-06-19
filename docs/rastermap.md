(rastermap_guide)=
# Rastermap

## Clustering: `n_clusters=None` vs specified values

- `n_clusters=None` or `0`: disables clustering, sorts individual neurons. This is only feasible for small datasets (~<200 neurons), as the sort becomes NP-hard for large N. Useful for fine-grained neuron sorting.
- `n_clusters=N` (e.g. 50–200): applies scaled k-means before sorting. Each cluster is treated as a “superneuron”. Default is 100. More clusters preserve detail but slow down sorting; fewer clusters compress more but risk mixing signals.

**Example:**  
- 54 neurons → `n_clusters=None`  
- Whole-brain recording (thousands) → `n_clusters=100`

---

## Clustering Step Details

1. **PCA** on (neurons × time) to extract `n_PCs=200` components → spatial basis.
2. **Clustering** in PCA space using scaled k-means → `U_nodes` (cluster centroids), `X_nodes` (average trace).
3. **Temporal basis** comes from the cluster traces (`X_nodes`).  
4. If `n_clusters=None`: each neuron becomes its own cluster.

This step compresses data from N neurons → k clusters and denoises by averaging.

---

## Superneurons and Binning

- A **superneuron** is the average of a group of similar neurons.
- After sorting, neurons can be binned into superneurons for display (e.g. 50-neuron bins).
- Binning is distinct from clustering: it’s for visualization.
- Smaller clusters than bin size are fine; bins may cross cluster boundaries.
- Superneurons smooth out noise and reduce display rows.

---

## Sorting via TSP + Segment Shifting

- Rastermap orders clusters using a variant of the Traveling Salesman Problem (TSP).
- Steps:
  1. Compute similarity matrix (`B B^T`) from cluster traces.
  2. Define a target matrix balancing global similarity and local continuity.
  3. Use heuristic segment shifting to optimize order (NP-hard problem).
  4. `verbose_sorting=True` shows shift steps.

**Locality parameter `w`** controls:
- `w=0`: prioritize global groupings.
- `w=1`: prioritize local sequences.

**Time lag**: allows asymmetric similarity (lead/lag in activity).

---

## Embedding and Upsampling

- After cluster ordering, neurons get assigned positions in 1D.
- `grid_upsample=10` creates more refined positions between clusters.
- Each neuron gets a fractional coordinate (`embedding`), not just a cluster label.
- Enables intra-cluster ordering based on small differences.

---

## Final Outputs

- `embedding`: fractional 1D coordinate per neuron.
- `embedding_clust`: cluster ID per neuron.
- `isort`: final sorted index list.
- Sorted similarity matrix is approximately block-diagonal.

---

## Summary: Model Step by Step

| Step                             | Compression      | Modeling Purpose                              |
|----------------------------------|------------------|-----------------------------------------------|
| PCA                              | Yes              | Reduces time dimensionality, extracts signals |
| Clustering → superneurons        | Yes              | Forms prototypes, reduces problem size        |
| TSP sorting                      | No               | Finds 1D manifold of similarity               |
| Segment shifting                 | No               | Optimizes the TSP ordering                    |
| Cluster upsampling               | Kind of          | Fine-grains neuron placement                  |
| Neuron Binning                   | Yes              | Smooths output for display                    |

---

## References

- [Rastermap: a discovery method for neural population recordings (Nature Neuroscience, 2024)](https://www.nature.com/articles/s41593-024-01783-4)
- [GitHub - MouseLand/rastermap](https://github.com/MouseLand/rastermap)
- Issues: [#36](https://github.com/MouseLand/rastermap/issues/36), [#38](https://github.com/MouseLand/rastermap/issues/38)
