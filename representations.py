"""
representations.py
==================

Visual / numerical representations of Eilenberg-MacLane coordinates as
points or vectors in concrete metric spaces (S^1, CP^N, wedge of circles,
R^n, ...).

Three layers, by increasing aggregation:

  1. Per-bar maps            : convert ONE bar (BG / MilEG element) into a
                               concrete numeric object (complex number,
                               unit C-vector, etc.).
  2. Per-cochain maps        : convert a WHOLE CechCochain valued in one
                               group into a CechCochain valued in another
                               group, by applying a per-bar map elementwise.
  3. Per-SheafSection maps   : convert an EM-coordinate SheafSection into
                               a point set in some R^d ready for plotting.


Table of contents
-----------------

  --- 1. Per-bar maps ---
  BZ_to_S1
  MilEG_S1_to_CPinf

  --- 2. Per-cochain maps ---
  cech_cochain_BZ_to_S1

  --- 3. Per-SheafSection maps ---
  Repr_BarB_Z_q_1_to_wedge_q_minus_1_circles

  --- 4. Metrics on bars ---
  distance_BG_bar_degree1
  distance_MilEG_bar
  distance_matrix_MilEG_bars     (fully vectorized; recommended for >100 bars)
  distance_MilBS1_bar
  distance_matrix_MilBS1_bars    (fully vectorized; recommended for >100 bars)
"""

import warnings
import numpy as np

import eg_tools
import cechcohomology as cc


# =============================================================================
# 1. Per-bar maps
# =============================================================================

def BZ_to_S1(bar):
    """
    Map a BG(Z) bar element to a point on the unit circle S^1.

    For a bar
        bar = {"t": [t_0, t_1, ..., t_n], "g": [g_0, g_1, ..., g_{n-1}]}
    with t_i nonnegative summing to 1 and g_k integers, return

        exp(2*pi*i * sum_{k=0..n-1}  (t_0 + t_1 + ... + t_k) * g_k)

    (note: t_n itself does not appear in the formula).

    Parameters
    ----------
    bar : dict
        A bar element. Must lie in BG(Z) — i.e., len(t) = len(g) + 1, the
        t_i nonnegative summing to 1, and every g_k an integer. This is
        verified via make_BG(Z)['in_group'].

    Returns
    -------
    complex
        A unit-modulus complex number on S^1.

    Raises
    ------
    ValueError
        If bar is not a valid element of BG(Z).
    """
    BG_Z = eg_tools.make_BG(eg_tools.Z)
    if not BG_Z['in_group'](bar):
        raise ValueError(
            f"BZ_to_S1: bar {bar} is not a valid element of BG(Z). "
            f"Required: len(t) = len(g)+1, t_i >= 0 summing to 1, g_k integers."
        )

    t = bar['t']
    g = bar['g']

    # Extra check: every g_k must really be an integer (in_group doesn't enforce this).
    if not all(isinstance(gk, (int, np.integer)) for gk in g):
        raise ValueError(
            f"BZ_to_S1: every g_k must be an integer (Z element); got g = {g}."
        )

    # Σ_{k=0..n-1} (t_0 + t_1 + ... + t_k) * g_k
    cumulative = 0.0
    total = 0.0
    for k in range(len(g)):
        cumulative += t[k]
        total += cumulative * g[k]

    return np.exp(2j * np.pi * total)


def MilEG_S1_to_CPinf(bar):
    """
    Map a MilEG(S^1) bar to a unit vector in C^n representing a point in CP^∞.

    Given a Milnor-style bar
        bar = {"t": [t_0,...,t_{n-1}], "g": [g_0,...,g_{n-1}]}
    with t_i in [0,1] summing to 1 (partition of unity) and g_i on S^1
    (complex with |g_i| = 1), return the unit vector

        v = (sqrt(t_0) * g_0, sqrt(t_1) * g_1, ..., sqrt(t_{n-1}) * g_{n-1})
              / ||·||₂

    Mathematically v already has unit norm (||v||² = Σ t_i |g_i|² = Σ t_i = 1);
    the explicit final normalization absorbs floating-point error.

    The class [v] ∈ CP^∞ (= S^∞ / S^1 action) is the actual target; this
    function returns the chosen representative in S^∞ ⊂ C^n.

    Parameters
    ----------
    bar : dict
        A MilEG-style bar: len(t) == len(g) == n,  t_i ≥ 0 sum to 1, g_i on S^1.

    Returns
    -------
    numpy.ndarray of complex, shape (n,)
        Unit vector representing a point in CP^∞.
    """
    t = bar['t']
    g = bar['g']

    if len(t) != len(g):
        raise ValueError(
            f"MilEG_S1_to_CPinf: input is a MilEG bar, so len(t) must equal len(g); "
            f"got len(t)={len(t)} and len(g)={len(g)}."
        )

    # v_i = sqrt(t_i) * g_i  (complex)
    v = np.array(
        [np.sqrt(t_i) * g_i for t_i, g_i in zip(t, g)],
        dtype=complex,
    )

    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError(
            f"MilEG_S1_to_CPinf: input vector has zero norm (all t_i are zero?). "
            f"Bar: {bar}"
        )

    return v / norm


# =============================================================================
# 2. Per-cochain maps
# =============================================================================

def cech_cochain_BZ_to_S1(eta):
    """
    Convert a BG(Z)-valued Cech cochain into an S^1-valued Cech cochain by
    applying `BZ_to_S1` elementwise to each (simplex, idx) value.

    The result has the same dim, cover, and section as eta, but its grp is
    replaced by eg_tools.S1.

    BZ_to_S1 itself validates per value that each input is a BG(Z) bar with
    integer g_k, so there is no need for a separate group-level check here.

    Parameters
    ----------
    eta : cechcohomology.CechCochain
        A Cech cochain whose grp is BG(Z) (i.e. eg_tools.make_BG(eg_tools.Z)).

    Returns
    -------
    cechcohomology.CechCochain
        A Cech cochain whose grp is eg_tools.S1.
    """
    def new_func(simplex):
        return {idx: BZ_to_S1(v) for idx, v in eta(simplex).items()}

    return cc.CechCochain(
        dim     = eta.dim,
        cover   = eta.cover,
        grp     = eg_tools.S1,
        func    = new_func,
        section = eta.section,
    )


# =============================================================================
# 3. Per-SheafSection maps
# =============================================================================

def Repr_BarB_Z_q_1_to_wedge_q_minus_1_circles(q, EiMacCoords):
    """
    Map a BG(Z_q)-valued 0-cocycle / SheafSection (EiMacCoords) into the 2D
    plane, drawing the image as a wedge of (q-1) lobes using the rose curve
        r = sin((q-1) * theta).

    For each index i in EiMacCoords.domain, the bar `EiMacCoords(i)` is
    classified by its degree (= len(bar['g'])):

      - degree 0 : identity bar  {t:[1], g:[]}.  Returns (0, 0).
      - degree 1 : non-trivial.  bar = {t:[t0, t1], g:[g0]} with g0 in Z_q.
            * if g0 == 0           -> (0, 0)
            * if g0 in {1, ..., q-1}, map to lobe `g0` via the angle
              interval described below; linearly interpolate within the
              interval according to t0 (t0=0 -> left endpoint,
              t0=1 -> right endpoint); then compute (x, y) from
              r = sin((q-1) * theta).
      - degree >= 2 : emits a warnings.warn with the offending index i and
            the bar, and returns (nan, nan).

    Angle intervals (one per non-zero label g0 in {1, ..., q-1}), regardless
    of q's parity:

          g0 = k -> theta in [ 2(k-1)*pi/(q-1),  (2k-1)*pi/(q-1) ]

    The (q-1) intervals are disjoint and separated by gaps of equal width
    (where sin((q-1)*theta) is negative). Together they tile [0, 2*pi)
    with alternating "lobe / gap" segments of length pi/(q-1) each.

    Parameters
    ----------
    q : int
        Must be a positive integer >= 2. The order of the cyclic group Z_q.
    EiMacCoords : sheaves.SheafSection
        A SheafSection over BG(Z_q): EiMacCoords(i) is a BG(Z_q) bar.

    Returns
    -------
    np.ndarray of shape (N, 2)
        Array of (x, y) coordinates, indexed in the same order as
        EiMacCoords.domain.
    """
    if not (isinstance(q, (int, np.integer)) and q >= 2):
        raise ValueError(f"q must be an integer >= 2, got {q}.")

    domain = list(EiMacCoords.domain)
    coords = np.empty((len(domain), 2), dtype=float)

    for n, i in enumerate(domain):
        bar = EiMacCoords(i)
        t   = bar['t']
        g   = bar['g']
        deg = len(g)

        if deg == 0:
            coords[n] = (0.0, 0.0)
            continue

        if deg >= 2:
            warnings.warn(
                f"Repr_BarB_Z_q_1_to_wedge_q_minus_1_circles: bar at index "
                f"i={i} has degree {deg} >= 2, which has no defined "
                f"representation in the wedge-of-circles model. "
                f"Bar = {bar}. Returning (nan, nan)."
            )
            coords[n] = (np.nan, np.nan)
            continue

        # ---- degree == 1 ----
        g0 = int(g[0]) % q   # canonicalize to {0, 1, ..., q-1}
        if g0 == 0:
            coords[n] = (0.0, 0.0)
            continue

        # Angle interval for lobe g0 (same formula for q even or odd):
        #     [ 2(g0-1) pi/(q-1),  (2 g0 - 1) pi/(q-1) ]
        left  = 2 * (g0 - 1) * np.pi / (q - 1)
        right = (2 * g0 - 1) * np.pi / (q - 1)

        t0    = float(t[0])
        theta = left + t0 * (right - left)
        r     = np.sin((q - 1) * theta)
        coords[n] = (r * np.cos(theta), r * np.sin(theta))

    return coords


# =============================================================================
# 4. Metrics on bars
# =============================================================================

def distance_BG_bar_degree1(bar1, bar2, G):
    """
    Distance between two degree-1 BG(G) bars

        bar1 = {'t': [t0, t1], 'g': [g0]},   bar2 = {'t': [s0, s1], 'g': [h0]}

    using
        N(bar)        = min{ 2*t0, 2*t1, d_G(g0, e) }
        d(bar1, bar2) = min{
                          d_G(g0, h0) + |t0 - s0| + |t1 - s1|,
                          N(bar1) + N(bar2)
                        }

    where d_G = G['distance'] and e = G['identity'].
    """
    for name, b in (('bar1', bar1), ('bar2', bar2)):
        if len(b['t']) != 2 or len(b['g']) != 1:
            raise ValueError(
                f"{name} must be a degree-1 BG bar (len(t)=2, len(g)=1); "
                f"got t={b['t']}, g={b['g']}."
            )

    t0, t1 = bar1['t']
    s0, s1 = bar2['t']
    g0 = bar1['g'][0]
    h0 = bar2['g'][0]
    d_G = G['distance']
    e   = G['identity']

    N1 = min(2.0 * t0, 2.0 * t1, d_G(g0, e))
    N2 = min(2.0 * s0, 2.0 * s1, d_G(h0, e))

    T_direct = d_G(g0, h0) + abs(t0 - s0) + abs(t1 - s1)
    T_via_id = N1 + N2

    return min(T_direct, T_via_id)


def _inner_bar_distance_BG_degree_le_1(a, b, G):
    """Helper: BG(G) distance between two inner bars, each of degree 0 or 1."""
    def is_identity(bar):
        return len(bar['t']) == 1 and len(bar['g']) == 0

    def N_deg1(bar):
        t0, t1 = bar['t']
        return min(2.0 * t0, 2.0 * t1, G['distance'](bar['g'][0], G['identity']))

    a_id, b_id = is_identity(a), is_identity(b)
    if a_id and b_id:
        return 0.0
    if a_id:
        return N_deg1(b)
    if b_id:
        return N_deg1(a)
    return distance_BG_bar_degree1(a, b, G)


def distance_MilEG_bar(bar1, bar2, G):
    """
    Fubini-Study-like distance between two MilEG bars:

        d(bar1, bar2) = arccos( sum_k sqrt(t_k * s_k) * cos( d_G(g_k, h_k) ) )

    where d_G is the BG(G) degree-≤1 distance on inner bars
    (handles identity inner bars).

    This is the per-pair function; for many bars use
    `distance_matrix_MilEG_bars` (vectorized) instead.
    """
    if len(bar1['t']) != len(bar2['t']):
        raise ValueError(
            f"bar1 / bar2 outer length mismatch: "
            f"{len(bar1['t'])} vs {len(bar2['t'])}."
        )
    if len(bar1['t']) != len(bar1['g']) or len(bar2['t']) != len(bar2['g']):
        raise ValueError("MilEG bars require len(t) == len(g).")

    s = 0.0
    for ti, gi, si, hi in zip(bar1['t'], bar1['g'], bar2['t'], bar2['g']):
        dg = _inner_bar_distance_BG_degree_le_1(gi, hi, G)
        s += np.sqrt(ti * si) * np.cos(dg)

    return float(np.arccos(np.clip(s, -1.0, 1.0)))


def distance_matrix_MilEG_bars(bars, G):
    """
    Vectorized pairwise distance matrix for a list of MilEG bars using

        d(x, y) = arccos( sum_k sqrt(t_k * s_k) * cos( d_G(g_k, h_k) ) ),

    where each inner g_k / h_k is a BG(G) bar of degree 0 or 1.

    Fast path: if G is eg_tools.Z (the typical case for MilEG bars produced
    by `dim1_cech_cocycle_to_MilB_G`), inner pairwise d_G is computed with
    pure numpy broadcasting (no Python loops in the inner double-loop).
    Other groups fall back to a slower per-element loop for d_G.

    Memory: builds (N, N, n_outer) float arrays. For N=1000, n_outer=6 this
    is about 50 MB per array (a few hundred MB total) — fits comfortably.

    Parameters
    ----------
    bars : list of dict
        N MilEG bars with uniform outer length n_outer.
    G : dict
        Group dict providing G['distance'], G['identity'].

    Returns
    -------
    np.ndarray (N, N), float
        Symmetric distance matrix in [0, pi], zeros on diagonal.
    """
    N = len(bars)
    if N == 0:
        return np.zeros((0, 0), dtype=float)

    n_outer = len(bars[0]['t'])
    for b in bars:
        if len(b['t']) != n_outer or len(b['g']) != n_outer:
            raise ValueError(
                "All MilEG bars must share the same outer length, and have "
                "len(t) == len(g)."
            )

    # ---- Extract outer t -> (N, n_outer) ----
    T = np.asarray([b['t'] for b in bars], dtype=float)

    # ---- Extract inner-bar arrays ----
    is_id    = np.zeros((N, n_outer), dtype=bool)
    inner_t0 = np.zeros((N, n_outer), dtype=float)
    inner_t1 = np.zeros((N, n_outer), dtype=float)
    inner_g0 = np.zeros((N, n_outer), dtype=float)

    for i, bar in enumerate(bars):
        for k, gi in enumerate(bar['g']):
            if len(gi['t']) == 1 and len(gi['g']) == 0:
                is_id[i, k] = True
            elif len(gi['t']) == 2 and len(gi['g']) == 1:
                inner_t0[i, k] = gi['t'][0]
                inner_t1[i, k] = gi['t'][1]
                inner_g0[i, k] = gi['g'][0]
            else:
                raise ValueError(
                    f"Inner bar at (i={i}, k={k}) is neither degree-0 identity "
                    f"nor degree-1: {gi}"
                )

    # ---- d_G(g0, identity) at every (i, k) ----
    fast_path_Z = (G is eg_tools.Z)
    identity = G['identity']

    if fast_path_Z:
        d_g_to_id = np.abs(inner_g0)
    else:
        d_g_to_id = np.zeros((N, n_outer), dtype=float)
        for i in range(N):
            for k in range(n_outer):
                if not is_id[i, k]:
                    d_g_to_id[i, k] = G['distance'](inner_g0[i, k], identity)

    # N(inner_bar) for degree-1 bars (meaningful only when is_id = False)
    N_inner = np.minimum(np.minimum(2.0 * inner_t0, 2.0 * inner_t1), d_g_to_id)

    # ---- Pairwise d_G(g0_i, g0_j) per position k: shape (N, N, n_outer) ----
    if fast_path_Z:
        d_gg = np.abs(inner_g0[:, None, :] - inner_g0[None, :, :])
    else:
        d_gg = np.zeros((N, N, n_outer), dtype=float)
        for i in range(N):
            for j in range(i, N):
                for k in range(n_outer):
                    if not is_id[i, k] and not is_id[j, k]:
                        d = G['distance'](inner_g0[i, k], inner_g0[j, k])
                        d_gg[i, j, k] = d
                        d_gg[j, i, k] = d

    d_t0 = np.abs(inner_t0[:, None, :] - inner_t0[None, :, :])
    d_t1 = np.abs(inner_t1[:, None, :] - inner_t1[None, :, :])

    direct_deg1 = d_gg + d_t0 + d_t1                                   # both deg-1
    N_sum_deg1  = N_inner[:, None, :] + N_inner[None, :, :]            # both deg-1
    both_deg1_d = np.minimum(direct_deg1, N_sum_deg1)

    # Masks for the four cases at each (i, j, k)
    is_id_i = is_id[:, None, :]
    is_id_j = is_id[None, :, :]
    both_id    = is_id_i & is_id_j
    only_i_id  = is_id_i & ~is_id_j
    only_j_id  = ~is_id_i & is_id_j

    # Case dispatch:
    #   both identity        -> 0
    #   only i identity      -> N_inner[j]
    #   only j identity      -> N_inner[i]
    #   both degree-1        -> min(direct, N_i + N_j)
    d_inner = np.where(both_id, 0.0,
              np.where(only_i_id, N_inner[None, :, :],
              np.where(only_j_id, N_inner[:, None, :],
                       both_deg1_d)))

    # ---- Outer formula: sum_k sqrt(t_k * s_k) * cos(d_inner_k) ----
    sqrt_T  = np.sqrt(T)
    sqrt_tt = sqrt_T[:, None, :] * sqrt_T[None, :, :]                 # (N, N, n_outer)
    cos_d   = np.cos(d_inner)
    inner_sum = np.sum(sqrt_tt * cos_d, axis=2)                       # (N, N)

    D = np.arccos(np.clip(inner_sum, -1.0, 1.0))
    np.fill_diagonal(D, 0.0)
    # symmetrize (defend against floating-point asymmetry)
    D = 0.5 * (D + D.T)
    return D



def distance_MilBS1_bar(bar1, bar2, normalize=True):
    """
    Milnor B(S^1) = CP^∞ 上两点的距离（S^1 对角作用的最优对齐已闭式解掉）。

    点 x = Σ_i t_i · g_i 落在 join S^1 * S^1 * ... 中，
    t_i ≥ 0 是 join 权重（理应 Σ=1），g_i ∈ S^1 是第 i 个 slot 的标签。

        d([x],[y]) = arccos( | Σ_i sqrt(t_i s_i) · g_i · conj(h_i) | )

    bar = {'t': [...], 'g': [...]}，两 list 等长；'g' 存【单位圆上的复数】 e^{iθ}。
      · 第 i 项是第 i 个 join slot；两 bar 按 slot 逐位对齐（短的补零权重）。
      · t_i=0 的 slot 自动因 sqrt(t_i s_i)=0 退出（正是 join 的 identification）。
    normalize=True 时把每个权重向量归一到 Σ=1，保证 |Z|≤1 严格成立。

    返回 ∈ [0, π/2]（取 |Z| 后 antipodal 方向被商掉，上界 π/2）。
    """
    t  = np.asarray(bar1['t'], dtype=float)
    s  = np.asarray(bar2['t'], dtype=float)
    g  = np.asarray(bar1['g'], dtype=complex)   # g_i = e^{iθ_i}
    h  = np.asarray(bar2['g'], dtype=complex)   # h_i = e^{iψ_i}

    # slot 对齐：短的一方补齐 —— 权重补 0，复数标签补 1（w=0 使其不贡献）
    n = max(len(t), len(s))
    pad_w = lambda a: np.concatenate([a, np.zeros(n - len(a))])       if len(a) < n else a
    pad_g = lambda a: np.concatenate([a, np.ones(n - len(a), complex)]) if len(a) < n else a
    t, s = pad_w(t), pad_w(s)
    g, h = pad_g(g), pad_g(h)

    if normalize:
        if t.sum() > 0: t = t / t.sum()
        if s.sum() > 0: s = s / s.sum()

    w = np.sqrt(t * s)                       # sqrt(t_i s_i)
    Z = np.sum(w * g * np.conj(h))           # Σ w_i · g_i · conj(h_i)
    return float(np.arccos(min(abs(Z), 1.0)))  # clip 防浮点越界


def distance_matrix_MilBS1_bars(bars, normalize=True):
    """
    Vectorized pairwise distance matrix for MilB(S^1) bars via the
    Fubini-Study distance on CP^∞:

        Z_{ij} = Σ_k sqrt(t_i[k] * t_j[k]) * g_i[k] * conj(g_j[k])
        d_{ij} = arccos( |Z_{ij}| )

    Rewritten as a single matrix multiplication by observing
        Z_{ij} = A_i · conj(A_j),   where   A_{ik} = sqrt(t_i[k]) * g_i[k].
    So  Z  =  A @ A.conj().T,  and  D  =  arccos( clip(|Z|, 0, 1) ).

    Bars of different lengths are padded to the max length with t=0, g=1
    (padded slots contribute nothing since sqrt(t·s)=0). If `normalize=True`
    each t-vector is renormalized to sum 1 before computing (matches
    `distance_MilBS1_bar` behaviour).

    Parameters
    ----------
    bars : list of dict
        Each with keys 't' (nonneg floats) and 'g' (unit-modulus complex).
        len(t) == len(g) per bar; lengths may differ across bars.
    normalize : bool, default True
        Normalize each t-vector to sum to 1 before computing.

    Returns
    -------
    np.ndarray (N, N), float
        Symmetric distance matrix in [0, π/2], zeros on diagonal.
    """
    N = len(bars)
    if N == 0:
        return np.zeros((0, 0), dtype=float)

    n = max(len(b['t']) for b in bars)   # padded slot count

    T = np.zeros((N, n), dtype=float)
    G = np.ones((N, n),  dtype=complex)  # pad g with 1 (no effect since padded t=0)

    for i, b in enumerate(bars):
        t = np.asarray(b['t'], dtype=float)
        g = np.asarray(b['g'], dtype=complex)
        if len(t) != len(g):
            raise ValueError(f"bar {i}: len(t) != len(g)  ({len(t)} vs {len(g)}).")
        T[i, :len(t)] = t
        G[i, :len(g)] = g

    if normalize:
        row_sums = T.sum(axis=1, keepdims=True)                # (N, 1)
        safe     = row_sums > 0
        T        = np.where(safe, T / np.where(safe, row_sums, 1.0), T)

    A = np.sqrt(T) * G                     # (N, n) complex; ||A_i||=1 when t sums to 1

    Z = A @ A.conj().T                     # (N, N) complex
    D = np.arccos(np.clip(np.abs(Z), 0.0, 1.0))

    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)                    # symmetrize floating-point residue
    return D