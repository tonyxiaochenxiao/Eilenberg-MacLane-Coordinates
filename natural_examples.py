"""
natural_examples.py
==========
"""

import numpy as np
import itertools
import plotly.graph_objects as go

import utils
import eg_tools
import cechcohomology as cc
import sheaves

def compute_EiMac_coords(
    eta_base_values,
    open_cover_filtration,
    group,
    data,
    centers,
    gamma1,
    geodesic_distance,
    dims_to_keep=(0, 1),     # Which dimensions of eta do you want to take record
):
    mark = "✓" if cc.VALIDATE else "?"
    banner = ("*** Running Eilenberg-MacLane Algorithm ***" if cc.VALIDATE
              else "*** Running Eilenberg-MacLane Algorithm (Fast - No validation) ***")
    print(banner)

    # Initialize eta
    print("Creating eta.")
    eta = make_eta(eta_base_values, open_cover_filtration[0], group)

    # Validating eta
    print("Validating eta:" if cc.VALIDATE else "Validating eta (skipped):")
    cc.validate_cech_cochain(eta)
    print(f"{mark} eta is a Čech {eta.dim}-cochain over F_{eta.grp['name']}.")
    cc.validate_cech_cocycle(eta)   # 我们的 eta 应该返回 True（因为四重交集为空，δη 是空的）
    print(f"{mark} eta is a Čech cocycle.")

    # Setting up g's
    print("→ Setting up bump functions g's")
    N_cover = len(open_cover_filtration[0])
    g = [None] + [
        (lambda idx, i=i: g_i(i, data[idx], centers, gamma1, geodesic_distance))
        for i in range(1, N_cover)
    ]
    print(f"· len(g) = {len(g)}  (g[0] is None placeholder)")

    etas_by_dim = {}
    if eta.dim in dims_to_keep:
        etas_by_dim[eta.dim] = eta
    while eta.dim > 0:
        # Dimension lowering
        print("")
        print("Dimension Lowering:")

        cc.validate_lifted_eta(eta)
        print(f"{mark} δ(lifted_eta[i]) == (eta | cover[i])")
        cc.validate_lifted_eta_2(eta)
        print(f"{mark} (lifted_eta[i] | U_i ∩ U_j) - (lifted_eta[j] | U_i ∩ U_j) is a cocycle on U_i ∩ U_j")

        print(f"→ Computing j_sharp(Q(i_sharp(eta))) [from dim {eta.dim} to dim {eta.dim-1}]")
        new_eta = cc.j_sharp(cc.validate_Q_lower_cech_on_F_EG(cc.i_sharp(eta), open_cover_filtration, g))
        print(f"· New Cech cochain's section is updated as: {new_eta.section}")
        cc.validate_cech_cocycle(new_eta)
        print(f"· New eta is a Cech {new_eta.dim}-cocycle")

        eta = new_eta
        if eta.dim in dims_to_keep:
            etas_by_dim[eta.dim] = eta

    print("→ Gluing eta into a classifying map")
    EiMacCoords = cc.zero_cocycle_to_sheaf_section(eta)

    print("")
    print("Eilenberg-MacLane coordinates is done:")
    print(f"· Domain (index): {EiMacCoords.domain[0]} ~ {EiMacCoords.domain[-1]}")
    print(f"· Codomain: {EiMacCoords.grp['name']}")
    return EiMacCoords, etas_by_dim

def Phi_i(i, x, small_radius, big_radius, centers, geodesic_distance): #i=0,1,2,...,N-1 (N=number of open sets); x is in R^3
    if geodesic_distance(centers[i],x) <= small_radius:
        return 1
    if geodesic_distance(centers[i],x) >= big_radius:
        return 0
    else:
        return (big_radius-geodesic_distance(centers[i],x))/(big_radius-small_radius)

def g_i(i, x, centers, radii, geodesic_distance):
    """
    Unified version of the old g1, g2, g3 functions.

    g_i(i, x, centers, radii) = min(
        max_{j = 0, 1, ..., i-1}  Phi_i(j, x, radii[i], radii[i-1], centers),
        Phi_i(i, x, radii[i], radii[0], centers)
    )

    Assumes radii is strictly decreasing so that radii[i] < radii[i-1] < radii[0],
    i.e. radii[i] is the "small radius" and radii[i-1] / radii[0] are the
    "big radius" in the corresponding Phi_i calls.

    Parameters
    ----------
    i : int, 1 <= i <= N - 1 (where N = len(open_cover))
        Index of this g_i.
    x : point in R^3
    centers : sequence of center points (one per open set in the cover)
    radii : list of floats, strictly decreasing
        radii[0] is the coarsest (largest balls); radii[i] is the current
        (smaller) radius.

    Returns
    -------
    float
    """
    if i < 1 or i >= len(radii):
        raise ValueError(
            f"g_i requires 1 <= i <= len(radii) - 1, but received i = {i} "
            f"with len(radii) = {len(radii)}."
        )

    inner_max  = max(Phi_i(j, x, radii[i], radii[i - 1], centers, geodesic_distance) for j in range(i))
    outer_term = Phi_i(i, x, radii[i], radii[0], centers, geodesic_distance)

    return min(inner_max, outer_term)

def nerve(dataset, centers, geodesic_distance, radius):
    """
    Compute the nerve statistics of balls U_i = {x in dataset : geodesic_distance(x, center_i) < radius}
    and print the sizes of all intersections.

    Fully generic: `dataset` and `centers` can be any iterables (numpy arrays,
    lists of dicts, lists of custom objects, etc.). The only requirement is
    that `geodesic_distance(x, c)` accepts a (dataset_item, center_item)
    pair and returns a nonnegative real.

    As an optional side-effect: if the inputs happen to coerce into float
    arrays of shape (N, 3) and (4, 3) (the original R³ + 4-center case),
    the function also draws a 3D plotly figure of the nerve. Otherwise
    no plot is produced.

    Parameters
    ----------
    dataset : iterable of items
    centers : iterable of items
    geodesic_distance : callable(item, item) -> nonnegative real
    radius : positive real

    Returns
    -------
    None.  Prints intersection counts; optionally draws a 3D figure.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")

    dataset_list = list(dataset)
    centers_list = list(centers)
    N = len(dataset_list)
    M = len(centers_list)

    # Build masks U_i — purely via geodesic_distance, no type assumptions
    masks = [
        np.array(
            [geodesic_distance(x, c) < radius for x in dataset_list],
            dtype=bool,
        )
        for c in centers_list
    ]

    # Print single-set counts
    print("Single sets:")
    for i in range(M):
        print(f"U({i}): {int(np.sum(masks[i]))}")

    # Compute and print all intersections of size >= 2
    intersection_counts = {}
    for r in range(2, M + 1):
        print(f"\n{r}-fold intersections:")
        for combo in itertools.combinations(range(M), r):
            mask = np.ones(N, dtype=bool)
            for i in combo:
                mask &= masks[i]
            count = int(np.sum(mask))
            intersection_counts[combo] = count
            combo_str = ",".join(map(str, combo))
            print(f"U({combo_str}): {count}")

    # Optional 3D plot: only if inputs look like float arrays in R³ with 4 centers
    try:
        centers_arr = np.asarray(centers_list, dtype=float)
        is_3d_4_centers = (
            centers_arr.ndim == 2
            and centers_arr.shape == (4, 3)
        )
    except (ValueError, TypeError):
        is_3d_4_centers = False

    if is_3d_4_centers:
        _nerve_plot_3d_4centers(centers_arr, masks, intersection_counts)


def _nerve_plot_3d_4centers(centers, masks, intersection_counts):
    """
    Draw a 3D plotly figure visualizing the nerve of a 4-center cover in R³:
      - centers as points (text label shows |U_i|)
      - edges labeled by pairwise intersection counts
      - face centroids labeled by triple intersection counts
      - tetrahedron centroid labeled by the 4-intersection count

    Internal helper for `nerve`. Assumes `centers` is a (4, 3) float array.
    """
    M = 4
    fig = go.Figure()

    # ===== Centers（点）=====
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='markers',
        marker=dict(size=7),
        name='Centers'
    ))

    # ===== 顶点编号（单独一层，保证显示）=====
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='text',
        text=[f"{i}: <b>{int(np.sum(masks[i]))}</b>" for i in range(M)],
        textposition='top center',
        showlegend=False,
        hoverinfo='skip'
    ))

    # ===== hover（显示 |Ui|）=====
    hover_text = [
        f"U{i}: {int(np.sum(masks[i]))}"
        for i in range(M)
    ]

    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

    # ===== 边 =====
    for i, j in itertools.combinations(range(M), 2):
        p, q = centers[i], centers[j]

        fig.add_trace(go.Scatter3d(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            z=[p[2], q[2]],
            mode='lines',
            line=dict(width=4),
            showlegend=False,
            hoverinfo='skip'
        ))

        mid = 0.5 * (p + q)
        pair_count = intersection_counts[(i, j)]

        fig.add_trace(go.Scatter3d(
            x=[mid[0]],
            y=[mid[1]],
            z=[mid[2]],
            mode='text',
            text=[f"{i}{j}: <b>{pair_count}</b>"],
            showlegend=False,
            hoverinfo='skip'
        ))

    # ===== 面 =====
    for combo in itertools.combinations(range(M), 3):
        pts = centers[list(combo)]
        centroid = np.mean(pts, axis=0)
        face_count = intersection_counts[combo]

        label = "".join(map(str, combo))  # 023

        fig.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='text',
            text=[f"{label}: <b>{face_count}</b>"],
            showlegend=False,
            hoverinfo='skip'
        ))

    # ===== 四交集 =====
    tetra_centroid = np.mean(centers, axis=0)
    total_count = intersection_counts[(0, 1, 2, 3)]

    fig.add_trace(go.Scatter3d(
        x=[tetra_centroid[0]],
        y=[tetra_centroid[1]],
        z=[tetra_centroid[2]],
        mode='text',
        text=[f"0123: <b>{total_count}</b>"],
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="Nerve of the cover",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        )
    )

    fig.show()

def compute_open_cover_filtration(data, centers, radii, geodesic_distance):
    """
    Compute a filtration of open covers, one for each radius in radii.

    Parameters
    ----------
    data : array-like, shape (N, d)
    centers : array-like, shape (M, d)
    radii : list of floats, non-increasing (radii[i] >= radii[i+1])
    geodesic_distance : function(x, y) -> float

    Returns
    -------
    filtration : list of numpy arrays (dtype=object)
        filtration[k] is the open cover computed with radius radii[k].
        Since radii is non-increasing, filtration[0] is the coarsest cover
        (largest balls) and filtration[-1] is the finest cover (smallest balls).
    """
    for i in range(len(radii) - 1):
        if radii[i] < radii[i + 1]:
            raise ValueError(
                f"radii must be non-increasing, but radii[{i}]={radii[i]} "
                f"< radii[{i+1}]={radii[i+1]}."
            )

    return [cc.compute_open_cover(data, centers, r, geodesic_distance) for r in radii]        


def make_eta(eta_base_values, cover, group, section=None):
    """
    Build a Čech cochain `eta` from a sparse `eta_base_values` dict.

    `eta_base_values` maps **sorted** simplex tuples (the "standard" form)
    to base values in `group`. Only the simplices you actually want to give
    a non-trivial value need to be listed — any simplex NOT in this dict is
    automatically treated as `group['identity']` (e.g. 0 for Z, 1 for S^1).
    This is the natural convention for cochains representing classes on a
    sub-complex of the full nerve.

    The dimension is inferred from the size of the keys (all keys must have
    the same length, equal to dim + 1).

    Alternating behaviour: for non-standard (unsorted) simplex input, the
    result is computed on the sorted form and `group['inv']` is applied iff
    the permutation is odd. Missing simplices return identity, and
    `inv(identity) = identity`, so the alternating rule is automatically
    consistent in that branch.

    Parameters
    ----------
    eta_base_values : dict
        Sparse map: sorted tuple of distinct ints -> base value in `group`.
        All keys must have the same length. Missing simplices default to
        `group['identity']`.
    cover : numpy array of dtype=object
        The open cover.
    group : dict
        Group dict in the eg_tools convention.
    section : array-like of ints or None, optional
        Local section.

    Returns
    -------
    cc.CechCochain
    """
    keys = list(eta_base_values.keys())
    if not keys:
        raise ValueError("eta_base_values must be non-empty so dim can be inferred.")

    # 推断维度
    dim = len(keys[0]) - 1
    # 检查其余是否同维
    all_same_dim = all(len(s) - 1 == dim for s in keys)
    if not all_same_dim:
        raise ValueError("eta_base_values has keys of inconsistent lengths.")

    identity = group['identity']

    def _eta_func(simplex):
        """
        Internal function for the cochain eta. Assumes simplex is a valid
        tuple of distinct ints (dim check and repeated-index check are done
        by CechCochain.__call__).

        Alternating: an odd permutation applies grp['inv'] to the base value.
        Simplices missing from `eta_base_values` return identity on every
        index of the intersection (and the intersection is typically empty
        for non-face simplices on a good cover, so this dict is usually {}).
        """
        sorted_simplex = tuple(sorted(simplex))
        indices = cc.compute_intersection(sorted_simplex, cover, section=section)

        if sorted_simplex not in eta_base_values:
            # Missing: identity on every covered index (no need to invert
            # for odd permutations because inv(identity) = identity).
            return {i: identity for i in indices}

        is_odd = cc._permutation_sign(simplex)
        base_value = eta_base_values[sorted_simplex]
        value = group['inv'](base_value) if is_odd else base_value
        return {i: value for i in indices}

    return cc.CechCochain(
        dim     = dim,
        cover   = cover,
        grp     = group,
        func    = _eta_func,
        section = section,
    )

from collections import Counter

def _bar_dim_signature(bar, depth):
    """
    Recursive helper: compute the dimension signature of a bar at the given
    nesting depth.

    - depth = 1 : bar is a BG bar (its g-list contains raw G elements).
                  Returns the integer  len(bar['g']).
    - depth = d (>= 2) : bar is a B^d G bar (its g-list contains B^{d-1} G
                  bars). Returns a tuple
                      ( signature_at_(d-1)(g_0), ..., signature_at_(d-1)(g_{n-1}) )
                  of length n = len(bar['g']).

    So depth=2 gives a flat tuple of ints, depth=3 gives a tuple of tuples,
    and so on. depth must be >= 1.
    """
    if depth == 1:
        return len(bar.get('g', []))
    return tuple(_bar_dim_signature(g_elem, depth - 1) for g_elem in bar.get('g', []))


def dimension_tuple_stats_on_sheaf_section(sheaf_section, depth, return_indices=False):
    """
    Tally the non-trivial dimension signatures of all values of a SheafSection
    whose entries are bars at nesting depth `depth` (i.e. elements of B^depth G).

    Trivial bars (those with empty outer g-list) are skipped — they typically
    dominate the count and aren't informative. The signature of a non-trivial
    bar is computed recursively (see _bar_dim_signature).

    Conventions per depth:
        depth = 0 : raw G elements, no bar structure — rejected (raises).
        depth = 1 : BG bar; signature is an integer = len(bar['g']).
        depth = 2 : BBG bar; signature is a tuple of ints (one per inner BG bar).
        depth = 3 : BBBG bar; signature is a tuple of tuples of ints.
        depth >= 4: similarly nested.

    Parameters
    ----------
    sheaf_section : sheaves.SheafSection
        Section whose values live in B^depth G.
    depth : int, >= 1
        Nesting depth of the bars.
    return_indices : bool, default False
        If True, also return a dict mapping each non-trivial signature to the
        list of indices that produced it (so you can look up the actual bars).

    Returns
    -------
    counts : collections.Counter
        Maps each non-trivial signature to the number of indices in
        sheaf_section.domain that produced it. Useful methods:
            .most_common(k)   top-k most frequent signatures
            sum(.values())    total non-trivial bars
    by_signature : dict[signature -> list of int]   (only if return_indices=True)
        Maps each non-trivial signature to the list of data indices that
        produced it, e.g. {1: [37, 412, ...], 2: [891, 1024]}.

    When return_indices is True, the function returns a 2-tuple
    (counts, by_signature). Otherwise it returns just `counts`.
    """
    if depth < 1:
        raise ValueError(
            f"dimension_tuple_stats: depth must be >= 1 (got {depth}). "
            f"depth=0 corresponds to raw G elements with no bar structure."
        )

    counts = Counter()
    by_signature = {} if return_indices else None

    for idx in sheaf_section.domain:
        idx_int = int(idx)
        bar = sheaf_section(idx_int)
        outer_g = bar.get('g', [])
        if not outer_g:
            continue   # outer trivial — skip
        signature = _bar_dim_signature(bar, depth)
        counts[signature] += 1
        if return_indices:
            by_signature.setdefault(signature, []).append(idx_int)

    if return_indices:
        return counts, by_signature
    return counts

def dimension_tuple_stats_on_cech_cochain(eta, depth, return_indices=False):
    """
    Per-simplex dimension-tuple statistics for a B^depth G -valued Čech cochain.

    For each standard eta.dim-simplex sigma, evaluate eta(sigma) to a dict
        {idx: bar}
    wrap it as a one-off sheaves.SheafSection (domain = dict keys, grp = eta.grp,
    func = lookup), and feed it through `dimension_tuple_stats_on_sheaf_section`.
    The result is a per-simplex Counter (optionally with indices).

    Parameters
    ----------
    eta : cc.CechCochain
        A Čech cochain whose values live in B^depth G (i.e. each eta(sigma)[idx]
        is a bar at nesting depth `depth`).
    depth : int, >= 1
        Nesting depth, same meaning as in dimension_tuple_stats_on_sheaf_section:
          depth=1  ->  BG bars,  signature is len(bar['g'])
          depth=2  ->  BBG bars, signature is tuple of inner BG dims
          depth=d  ->  B^d G bars, signature is tuple of depth-(d-1) signatures
    return_indices : bool, default False
        If True, also return a per-simplex map from signature to the list of
        indices that produced it.

    Returns
    -------
    results : dict[simplex -> collections.Counter]
        Maps each standard simplex (a tuple of ints) to a Counter of its
        non-trivial dimension signatures. Simplices with no non-trivial bars
        produce an empty Counter (still included in the dict for completeness).
    indices : dict[simplex -> dict[signature -> list of int]]
        (only when return_indices=True) Maps each simplex to a dict mapping
        each non-trivial signature to the list of indices that produced it.

    When return_indices is True, the function returns a 2-tuple
    (results, indices). Otherwise it returns just `results`.

    Examples
    --------
        # eta is a BG-valued 1-cochain
        per_simplex = dimension_tuple_stats_on_cech_cochain(eta, depth=1)
        for sigma, counts in per_simplex.items():
            if counts:
                print(sigma, counts.most_common(5))

        # With indices: see which specific indices are non-trivial per simplex
        per_simplex, per_simplex_indices = dimension_tuple_stats_on_cech_cochain(
            eta, depth=1, return_indices=True,
        )
        for sigma, by_sig in per_simplex_indices.items():
            for sig, idxs in by_sig.items():
                print(sigma, "signature", sig, "->", idxs)
    """
    N = len(eta.cover)
    results = {}
    indices = {} if return_indices else None

    for simplex in itertools.combinations(range(N), eta.dim + 1):
        # Materialize eta(simplex) as a plain dict with Python-int keys,
        # then wrap as a SheafSection so we can reuse the sheaf-section version.
        d_norm = {int(k): v for k, v in eta(simplex).items()}
        local_section = sheaves.SheafSection(
            grp    = eta.grp,
            func   = lambda idx, _d=d_norm: _d[int(idx)],
            domain = list(d_norm.keys()),
        )
        if return_indices:
            counts, by_sig = dimension_tuple_stats_on_sheaf_section(
                local_section, depth, return_indices=True,
            )
            results[simplex] = counts
            indices[simplex] = by_sig
        else:
            results[simplex] = dimension_tuple_stats_on_sheaf_section(local_section, depth)

    if return_indices:
        return results, indices
    return results


def dim1_cech_cocycle_to_MilB_G(eta, centers, radius, data, geodesic_distance):
    """
    Convert a 1-dimensional Čech cocycle to a list of MilG (Milnor's EG)
    elements, one per data index.
    
    This is Jose Perea's work on H^1(B,F_G) ≌ Prin_G(B) ≌ [B,BG]

    For each data index m in {0, 1, ..., len(data)-1}:
      - Find any k such that m ∈ eta.cover[k]. If no such k exists, set
        list[m] = NaN and collect m for a single batched warning at the end.
      - Otherwise, let b = data[m] and build the MilG element

            {  "t": [P(0, b), P(1, b), ..., P(N-1, b)],
               "g": [g_0, g_1, ..., g_{N-1}]  }

        where
            P(i, b) =      relu(radius - dist(centers[i], b))
                         / sum_j  relu(radius - dist(centers[j], b))

        and
            g_i =  eta.grp['identity']             if i == k
                                                    (avoids repeated-index eta call)
            g_i =  eta((k, i))[m]                   if i != k  and  m ∈ U_i
            g_i =  eta.grp['identity']             if i != k  and  m ∉ U_i
                                                    (P(i, b) = 0 anyway, so g_i is irrelevant)

    Note this is the Milnor (join) realization of EG, NOT the bar realization
    used elsewhere in eg_tools. In particular the resulting bar always has
    length N (one entry per center), with the convention that g_i is
    irrelevant when t_i = 0. We deliberately fill those with
    eta.grp['identity'] (rather than NaN) so downstream operations stay
    well-defined even if they don't know about the t_i = 0 convention.

    Parameters
    ----------
    eta : cc.CechCochain
        A 1-dimensional Čech cocycle.
    centers : array-like of points
        The N cover centers in the ambient space.
    radius : float
        Cover radius (uniform across all U_i).
    data : array-like of points, length M
        The dataset.
    geodesic_distance : callable(x, y) -> float
        Distance in the ambient space.

    Returns
    -------
    list of length len(data)
        Each entry is either a dict {"t": [...], "g": [...]} (a MilG element)
        or float('nan') for data points not covered by any U_k.
    """
    # ---- Input checks ----
    if eta.dim != 1:
        raise ValueError(
            f"dim1_cech_cocycle_to_MilG requires eta.dim == 1, got {eta.dim}."
        )

    if eta.section is not None:
        warnings.warn(
            "dim1_cech_cocycle_to_MilG: eta.section is not None. The algorithm "
            "expects a cocycle on the global section (eta.section = None). "
            "If eta.section happens to equal the union of all eta.cover[k], "
            "the output is still correct; otherwise it may be incomplete."
        )

    N = len(centers)
    M = len(data)
    identity = eta.grp['identity']

    # ---- Precomputations ----
    # cover_sets[k] = set of int indices in eta.cover[k], for fast membership tests
    cover_sets = [set(int(j) for j in eta.cover[k]) for k in range(N)]

    # Pre-evaluate eta on every standard / non-standard pair (k, i) with k != i.
    # eta((k, i))[idx] is the directed value k -> i at idx; our CechCochain
    # handles the alternating sign automatically when k > i. We cache all
    # N*(N-1) such dicts up front so the per-index loop is just dict lookup.
    eta_cache = {
        (k, i): eta((k, i))
        for k in range(N)
        for i in range(N)
        if k != i
    }

    # ---- Main loop ----
    result = [None] * M
    missing = []

    for index in range(M):
        # Find any k such that index ∈ U_k
        k_found = None
        for k in range(N):
            if index in cover_sets[k]:
                k_found = k
                break

        if k_found is None:
            missing.append(index)
            result[index] = float('nan')
            continue

        k = k_found
        b = data[index]

        # Partition of unity at b: relu(radius - d_i) normalized
        relus = [max(radius - geodesic_distance(centers[i], b), 0.0) for i in range(N)]
        denom = sum(relus)
        if denom > 0:
            t = [r / denom for r in relus]
        else:
            # Should not happen here because we found a k with index ∈ U_k,
            # which means dist(centers[k], b) < radius, so relus[k] > 0.
            t = [0.0] * N

        # Build g
        g = []
        for i in range(N):
            if i == k:
                g.append(identity)
            elif index in cover_sets[i]:
                g.append(eta_cache[(k, i)][index])
            else:
                # P(i, b) = 0 in this case; g_i is irrelevant. Fill identity
                # to keep the bar formally well-defined.
                g.append(identity)

        result[index] = {"t": t, "g": g}

    # ---- One batched warning for missing indices ----
    if missing:
        preview = missing[:20]
        more = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
        warnings.warn(
            f"dim1_cech_cocycle_to_MilG: {len(missing)} data index/indices not "
            f"covered by any U_k. Their entries in the result list are NaN. "
            f"First few missing indices: {preview}{more}."
        )

    return result
