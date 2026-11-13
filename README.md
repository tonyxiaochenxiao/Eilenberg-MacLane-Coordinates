# Eilenberg–MacLane Coordinates

**From persistent cohomology to explicit maps into arbitrary Eilenberg–MacLane spaces $K(G,n)$.**

This repository contains our Python implementation of the **Sparse Eilenberg–MacLane Algorithm** from

> T. X. Xiao and J. A. Perea. *From Persistent Cohomology to Maps into Arbitrary Eilenberg–MacLane Spaces.* Preprint, 2026.

Given a dataset $X$ and a cohomology class $\eta \in H^n(-;G)$ detected on it, we construct an explicit coordinate
$$f_\eta : X^{(\alpha)} \longrightarrow K(G,n),$$
defined on the whole $\alpha$-thickening $X^{(\alpha)}$ of the data, for **every degree $n \geq 1$ and every discrete abelian group $G$**.

---

## Why Eilenberg–MacLane coordinates?

A persistence barcode tells us that a cohomology class *exists*; it does not tell us *where it lives* in the data. Representability of cohomology,
$$H^n(B;G) \cong [B, K(G,n)],$$
says every class is realized by a map into $K(G,n)$, but this bijection is purely existential. Over the past fifteen years, explicit coordinates have been constructed case by case, each time for a target space that happens to have a concrete low-dimensional model:

| Coordinate type | Cohomology source | Target | Concrete model |
|---|---|---|---|
| Circular | $PH^1(X;ℤ)$ | $K(ℤ,1)$ | $S^1$ |
| Toroidal | $PH^1(X;ℤ^l)$ | $K(ℤ^l,1)$ | $T^l$ |
| Real projective | $PH^1(X;ℤ_2)$ | $K(ℤ_2,1)$ | $ℝℙ^\infty$ |
| Lens | $PH^1(X;ℤ_q)$ | $K(ℤ_q,1)$ | $S^\infty/ℤ_q$ |
| Complex projective | $PH^2(X;ℤ)$ | $K(ℤ,2)$ | $ℂℙ^\infty$ |
| **Eilenberg–MacLane (this work)** | $PH^n(X;G)$ | $K(G,n)$ | $B^nG$ (iterated bar construction) |

We replace this case-by-case approach with a single framework that works uniformly in all degrees and for all discrete abelian coefficient groups. In particular, the code computes coordinate types that no previous method computes, such as $K(ℤ_2,2)$-coordinates on $ℝℙ^2$.

## How it works

We model $K(G,n)$ by the $n$-fold bar construction $B^nG$, which is again a topological abelian group at every stage. The universal bundles $B^{k}G \to EB^{k}G \to B^{k+1}G$ give a chain of connecting isomorphisms in Čech cohomology
$$\check H^n(B;\mathcal F_G) \cong \check H^{n-1}(B;\mathcal F_{BG}) \cong \cdots \cong \check H^1(B;\mathcal F_{B^{n-1}G}) \cong [B, B^nG],$$
which lowers the cohomological degree one bar construction at a time.

The core of the paper, and of this code, is the **$Q^n$-Algorithm**: an explicit, constructive inverse of the coboundary on sheaves of maps into a contractible group $E$. Given a Čech cocycle $\eta \in \check Z^n(\mathcal U;\mathcal F_E)$ on a finite cover by balls, it builds a cochain $Q^n(\eta)$ with $d^{\,n-1}Q^n(\eta) = \eta$ by
1. **lifting** $\eta$ locally on each open set,
2. **extending** the local correction terms with the contracting homotopy of $E$ and piecewise-linear bump functions, and
3. **gluing** the pieces together, one open set at a time, along a shrinking filtration of the cover.

No partition of unity, optimization, or global linear solve is needed. Calling $Q^n, Q^{n-1}, \dots, Q^1$ in succession along the tower $G, BG, \dots, B^{n-1}G$ yields the coordinate $f_\eta$.

We implement two versions of the final step:

- **Canonical pipeline.** Apply $Q^1$ to obtain $f_\eta : X^{(\alpha)} \to B^nG$ directly.
- **Milnor variant.** Stop at a Čech 1-cocycle and pass to its classifying map into Milnor's model $\mathbf B(B^{n-1}G)$ via a partition of unity. For $n=1$ this recovers Perea's sparse circular coordinates. Whenever $B^{n-1}G$ carries a metric, the target inherits one, so Isomap, MDS and Ripser apply directly to the coordinates.

## Installation

We ran all experiments with the following versions, which `requirements.txt` pins exactly:

| Package | Version | Package | Version |
|---|---|---|---|
| Python | 3.9.13 | ripser | 0.6.4 |
| numpy | 1.23.5 | persim | 0.3.1 |
| scipy | 1.11.3 | numba | 0.56.4 |
| scikit-learn | 1.6.1 | plotly | 5.9.0 |
| matplotlib | 3.8.0 | DREiMac | 0.3.0 (bundled) |

```bash
git clone https://github.com/tonyxiaochenxiao/Eilenberg-MacLane-Coordinates.git
cd Eilenberg-MacLane-Coordinates

conda create -n emcoords python=3.9.13
conda activate emcoords
pip install -r requirements.txt
```

The notebook also uses [DREiMac](https://github.com/scikit-tda/DREiMac) to sample datasets such as `GeometryExamples.sphere`. So that future DREiMac releases cannot break the code, we ship the exact version we used, **DREiMac 0.3.0** (Apache 2.0), in `DREiMac/dreimac/`. You do not need to install it. The first cell of the notebook puts the bundled copy on the path:

```python
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "DREiMac"))
from dreimac import GeometryExamples
```

Launch Jupyter from the repository root so that this path resolves correctly.

## Quick start: a circular coordinate on $S^1$

The example below runs the canonical pipeline for the generator of $H^1(S^1;ℤ)$ on a cover of the circle by three geodesic balls (Experiment 1 of the paper). Run it from the repository root.

```python
import numpy as np
import eg_tools
import cechcohomology as cc
import natural_examples as exn
import representations as rep

cc.VALIDATE = False          # skip the (slow) internal consistency checks

# 1. Data: points on the unit circle
rng = np.random.default_rng(0)
theta = 2 * np.pi * rng.random(300)
data = np.stack([np.cos(theta), np.sin(theta)], axis=1)

def geodesic_distance(x, y):
    return np.arccos(np.clip(x[0] * y[0] + x[1] * y[1], -1.0, 1.0))

# 2. Cover by three balls centred at the cube roots of unity,
#    with a strictly decreasing sequence of radii (the cover filtration)
centers = [[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]]
alpha = geodesic_distance(centers[0], centers[1])     # largest good-cover radius
gamma = [alpha, 1.6, 1.2]
open_cover_filtration = exn.compute_open_cover_filtration(
    data, centers, gamma, geodesic_distance)

# 3. The Čech 1-cocycle eta_01 = 1, eta_02 = eta_12 = 0 (generator of H^1(S^1; Z))
eta_base_values = {(0, 1): 1, (0, 2): 0, (1, 2): 0}

# 4. Run the Eilenberg–MacLane Algorithm
EiMacCoords, etas_by_dim = exn.compute_EiMac_coords(
    eta_base_values, open_cover_filtration, eg_tools.Z,
    data, centers, gamma, geodesic_distance, dims_to_keep=(0, 1))

# EiMacCoords(i) is a point of BZ, written as a bar {"t": [...], "g": [...]}
print(EiMacCoords(0))

# Convert BZ ≃ S^1 to unit complex numbers
z = np.array([rep.BZ_to_S1(EiMacCoords(i)) for i in range(len(data))])
```

To compute a coordinate in higher degree, pass a degree-$n$ cocycle: for example, a dictionary keyed by triples $(i_0,i_1,i_2)$ for $n=2$. The same call then lowers the degree $n$ times and returns a $B^nG$-valued coordinate. To change the coefficient group, pass a different group: `eg_tools.Z`, or `eg_tools.make_Zq(q)` for $ℤ_q$.

## Reproducing the experiments

`E-M_Coordinate_Computation_Newest.ipynb` reproduces all five experiments of Section 5 of the paper, in order.

| Notebook section | Paper | What it computes |
|---|---|---|
| $S^1$, $Bℤ$-coordinates | Experiment 1 | The canonical $Bℤ$-coordinate of a noisy circle from the generator of $H^1(S^1;ℤ)$, on a cover by three geodesic balls. We realize it in $S^1$ and check it with Isomap. |
| $S^1$, $\mathbf Bℤ$-coordinates | Experiment 1 | The same class through the Milnor variant. For $n=1$ this is the classical circular coordinate, so the two pipelines can be compared directly. |
| $S^1$, $Bℤ_q$-coordinates | Experiment 2 | A $Bℤ_5$-coordinate from the same data, drawn in the 1-skeleton $\bigvee_{i=0}^{q-1} S^1$ of $L^\infty_q$. Comparing it with the $Bℤ$-coordinate shows why Bockstein lifting matters. |
| $S^2$ (with local density), $B(Bℤ)$-coordinates | Experiment 3 | A $B^2ℤ$-coordinate from the generator of $H^2(S^2;ℤ)$ on a tetrahedral cover by four balls, with extra samples where the bump functions are nontrivial. The canonical pipeline iterates $Q^2$ and $Q^1$. We plot the result through its barycentric coordinates $(t_0,t_1)$ and map it back to $S^2$ by stereographic projection. |
| $S^2$, $B^2ℤ$-coordinates | Experiment 4 | The Milnor variant applied to the degree-1 cocycle from the previous section, realized in $ℂℙ^\infty$ with the Fubini–Study metric. Isomap plus Ripser recover the persistent cohomology of $S^2$. |
| $ℝℙ^2$, $\mathbf B(Bℤ_2)$-coordinates | Experiment 5 | A **new coordinate type**: a $K(ℤ_2,2)$-coordinate from the generator of $H^2(ℝℙ^2;ℤ_2)$, on the 6-vertex minimal triangulation with the simplicial $\ell^1$ metric. Isomap recovers the Roman surface, and Ripser recovers the cohomology of $ℝℙ^2$ in both $ℤ_2$ and $ℤ_3$ coefficients. |

Each section builds the cover filtration and the cocycle, runs `compute_EiMac_coords` (or the Milnor variant `dim1_cech_cocycle_to_MilB_G`), computes a distance matrix on the resulting coordinates, and then runs Isomap and Ripser on it.

## Repository structure

| File | Contents |
|---|---|
| `natural_examples.py` | Main entry point `compute_EiMac_coords`, plus cover filtrations (`compute_open_cover_filtration`), bump functions `g_i`, cocycle construction (`make_eta`), the Milnor variant (`dim1_cech_cocycle_to_MilB_G`), and nerve plots. |
| `cechcohomology.py` | Čech cochains on the nerve of a cover (`CechCochain`): coboundary, lifting, extension by softness, gluing, the maps $i_\sharp$ and $j_\sharp$, and the $Q^n$-Algorithm `Q_lower_cech_dim_on_F_EG`. |
| `eg_tools.py` | Groups and the bar construction: $ℤ$, $S^1$, $ℤ_q$ (`make_Zq`), $EG$ (`make_EG`), $BG$ (`make_BG`), the group law on bars, and the contracting homotopy of $EG$ (`homotopy_eg`). |
| `sheaves.py` | `SheafSection`, a section of $\mathcal F_G$ over a set of data indices. |
| `representations.py` | Concrete realizations and metrics: $Bℤ \to S^1$, $\mathbf B S^1 \to ℂℙ^\infty$, $Bℤ_q \to$ wedge of circles, and distance matrices on bars, including the Fubini–Study metric on $\mathbf B S^1$. |
| `utils.py`, `math_tools_miscellaneous.py` | Helper maps: Hopf map, Riemann sphere, projective PCA. |
| `E-M_Coordinate_Computation_Newest.ipynb` | Notebook reproducing Experiments 1–5 of the paper. |
| `requirements.txt` | Exact versions of all dependencies. |
| `DREiMac/` | Bundled copy of DREiMac 0.3.0 (Apache 2.0), used by the notebook. |

### Data conventions

- **Groups** are dictionaries with keys `name`, `grplaw`, `identity`, `inv`, `eq`, and, for $ℤ$, $S^1$ and $ℤ_q$, a bi-invariant `distance`.
- **Bars**, i.e. points of $EG$ or $BG$, are dictionaries `{"t": [t_0, ..., t_n], "g": [g_0, ..., g_n]}` of barycentric coordinates and group elements. The groups $EG$ and $BG$ are themselves group dictionaries, so the construction iterates: $B(BG)$, $B^3G$, and so on.
- **Čech cochains** are `CechCochain` objects carrying their own dimension, cover, coefficient group and domain. All functions in `cechcohomology.py` take and return `CechCochain`s.
- **Validation.** `cechcohomology.VALIDATE = True` (the default) checks the cocycle and coboundary identities at every step of the algorithm. Set it to `False` for speed once your setup is working.

## Current scope

The implementation takes as input a Čech cocycle on the nerve of a finite cover of the data by balls, specified through `eta_base_values`. In the paper's pipeline this cocycle comes from a persistent cohomology class via Ripser and Bockstein lifting (Sections 3.1–3.2). That front end is not yet automated in this repository.

## Citation

If you use this code, please cite:

```bibtex
@misc{XiaoPerea2026EM,
  author = {Xiao, Tony Xiaochen and Perea, Jose A.},
  title  = {From Persistent Cohomology to Maps into Arbitrary {E}ilenberg--{M}ac{L}ane Spaces},
  year   = {2026},
  note   = {Preprint}
}
```

## Acknowledgments

The example notebook uses [DREiMac](https://github.com/scikit-tda/DREiMac) (J. A. Perea, L. Scoccola, C. J. Tralie et al.), [Ripser](https://github.com/scikit-tda/ripser.py), and [persim](https://github.com/scikit-tda/persim).

## Contact

Tony Xiaochen Xiao, Department of Mathematics, Northeastern University. Website: <https://tonyxiaochenxiao.github.io/>
