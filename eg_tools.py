import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from numbers import Number

# =============================================================================
# Internal utility
# =============================================================================

def _eq(a, b, tol=1e-12):
    """
    Low-level equality test: tolerance-based for numbers, strict `==` otherwise.

    NOTE
    ----
    This is the primitive "number-or-strict" equality used inside this module
    as a building block (e.g. for comparing partition-of-unity weights t_i to 0).
    It is also the underlying implementation of `Z['eq']`, `S1['eq']`,
    `make_Zq(q)['eq']` — primitive groups whose elements ARE numbers.

    DO NOT use `_eq` to compare group elements of non-primitive groups
    (e.g. EG / BG bars), because for non-numbers it falls back to Python `==`,
    which is strict dict equality and fails on floating-point differences.
    For comparing arbitrary group elements, always use the group's own `eq`
    function (e.g. `G['eq']`), which is tolerance-aware by construction.
    """
    if isinstance(a, Number) and isinstance(b, Number):
        return abs(a - b) <= tol
    return a == b

# =============================================================================
# Group definitions
# Each group G is a dict with keys:
#   'name'     : str, human-readable label
#   'source'   : the object this group was derived from, or None for
#                "primitive" (non-derived) groups. For make_Zq(q) this is q;
#                for make_EG(G) this is the input group G.
#   'in_group' : callable(x) -> bool, membership test
#   'grplaw'   : callable(a, b) -> element, binary group operation
#   'identity' : element, the identity element of G
#   'inv'      : callable(x) -> element, group inverse
#   'eq'       : callable(a, b, tol=1e-12) -> bool, equality test with tolerance
#   'distance' : callable(a, b) -> float, bi-invariant metric on G
#                (only on primitive groups Z, S^1, Z_q for now)
# =============================================================================

# Integer addition group (Z, +)
Z = {
    'name'     : 'Z',
    'source'   : None,
    'in_group' : lambda x: isinstance(x, (int, np.integer)),
    'grplaw'   : lambda a, b: a + b,
    'identity' : 0,
    'inv'      : lambda x: -x,
    'eq'       : lambda a, b, tol=1e-12: _eq(a, b, tol),
    'distance' : lambda a, b: abs(int(a) - int(b)),
}

# Unit circle group (S^1, *), elements are complex numbers with |z| = 1
S1 = {
    'name'     : 'S^1',
    'source'   : None,
    'in_group' : lambda x, tol=1e-9: isinstance(x, Number) and abs(abs(x) - 1.0) <= tol,
    'grplaw'   : lambda a, b: a * b,
    'identity' : 1+0j,
    'inv'      : lambda x: complex(x).conjugate(),
    'eq'       : lambda a, b, tol=1e-12: _eq(a, b, tol),
    'distance' : lambda a, b: float(np.arccos(
        np.clip((complex(a).conjugate() * complex(b)).real, -1.0, 1.0)
    )),
}

def make_Zq(q):
    """
    Return the cyclic group Z/qZ (integers mod q under addition) as a group dict.

    Parameters
    ----------
    q : int
        The modulus. Must be a positive integer.

    Returns
    -------
    dict with keys: name, in_group, grplaw, identity, inv, eq.
    """
    if not isinstance(q, int) or q <= 0:
        raise ValueError(f"q must be a positive integer, got {q}.")
    def _zq_distance(a, b):
        d = abs(int(a) % q - int(b) % q)
        return min(d, q - d)
    return {
        'name'     : f'Z_{q}',
        'source'   : q,
        'in_group' : lambda x: isinstance(x, (int, np.integer)) and 0 <= int(x) < q,
        'grplaw'   : lambda a, b: int((a + b) % q),
        'identity' : 0,
        'inv'      : lambda x: int((-x) % q),
        'eq'       : lambda a, b, tol=1e-12: _eq(a, b, tol),
        'distance' : _zq_distance,
    }

def make_EG(G):
    """
    Construct the group EG (the total space of the universal principal G-bundle)
    as a group dict, given a group G.

    Elements of EG are bar elements: dicts {"t": [...], "g": [...]} where
        t = [t0, ..., tn] with ti >= 0 and sum(ti) = 1
        g = [g0, ..., gn] are elements of G

    Parameters
    ----------
    G : dict
        A group dict with keys: name, in_group, grplaw, identity, inv, eq.

    Returns
    -------
    dict with keys: name, in_group, grplaw, identity, inv, eq.

    Notes
    -----
    'eq' is defined as exact Python equality (bar1 == bar2) on the bar dicts.
    EG does not carry a natural metric, so no tolerance-based comparison is used.
    The tol parameter in 'eq' is accepted for interface consistency but ignored.
    """
    def _in_EG(bar):
        try:
            _validate_eg_element(bar["t"], bar["g"])
            return True
        except Exception:
            return False

    def _eq_EG(bar1, bar2, tol=1e-12):
        """
        Element-wise tolerance-based equality on EG bars. Compares the t lists
        with float tolerance and the g lists using G's own eq. Assumes both
        inputs are in reduced form (eg_group_law produces reduced output).
        """
        t1, t2 = bar1['t'], bar2['t']
        g1, g2 = bar1['g'], bar2['g']
        if len(t1) != len(t2) or len(g1) != len(g2):
            return False
        if not all(abs(a - b) <= tol for a, b in zip(t1, t2)):
            return False
        if not all(G['eq'](a, b, tol) for a, b in zip(g1, g2)):
            return False
        return True

    return {
        'name'     : f'EG({G["name"]})',
        'source'   : G,
        'in_group' : _in_EG,
        'grplaw'   : lambda bar1, bar2: eg_group_law(bar1, bar2, grplaw=G['grplaw'], identity=G['identity'], eq=G['eq']),
        'identity' : {"t": [1], "g": [G['identity']]},
        'inv'      : lambda bar: eg_inv(bar, inv=G['inv'], grplaw=G['grplaw'], identity=G['identity'], eq=G['eq']),
        'eq'       : _eq_EG,
    }

def make_BG(G):
    """
    Construct the group BG (the classifying space of G, here as the quotient
    EG/G) as a group dict, given a group G.

    Elements of BG are bar elements: dicts {"t": [...], "g": [...]} where
        t = [t0, ..., tn] with ti >= 0 and sum(ti) = 1
        g = [g0, ..., g_{n-1}] are elements of G
    Note len(t) = len(g) + 1 (one more t than g), in contrast to EG where the
    lengths are equal.

    The group law and inverse are inherited from EG via the round-trip
        BG -- BG_to_EG --> EG -- group law / inv --> EG -- EG_to_BG --> BG
    The conversions are: BG_to_EG appends G's identity to the g-list; EG_to_BG
    drops the last g-entry.

    Parameters
    ----------
    G : dict
        A group dict with keys: name, source, in_group, grplaw, identity, inv, eq.

    Returns
    -------
    dict with keys: name, source, in_group, grplaw, identity, inv, eq.

    Notes
    -----
    'eq' is exact Python equality on the bar dicts (no metric on BG).
    """
    def _in_BG(bar):
        try:
            _validate_bg_element(bar["t"], bar["g"])
            return True
        except Exception:
            return False

    def _eq_BG(bar1, bar2, tol=1e-12):
        """
        Element-wise tolerance-based equality on BG bars. Compares the t lists
        with float tolerance and the g lists using G's own eq. Assumes both
        inputs are in reduced form (the BG group law produces reduced output
        via EG_to_BG ∘ eg_group_law ∘ BG_to_EG).
        """
        t1, t2 = bar1['t'], bar2['t']
        g1, g2 = bar1['g'], bar2['g']
        if len(t1) != len(t2) or len(g1) != len(g2):
            return False
        if not all(abs(a - b) <= tol for a, b in zip(t1, t2)):
            return False
        if not all(G['eq'](a, b, tol) for a, b in zip(g1, g2)):
            return False
        return True

    return {
        'name'     : f'BG({G["name"]})',
        'source'   : G,
        'in_group' : _in_BG,
        'grplaw'   : lambda bar1, bar2: EG_to_BG(
            eg_group_law(
                BG_to_EG(bar1, G['identity']),
                BG_to_EG(bar2, G['identity']),
                grplaw=G['grplaw'], identity=G['identity'], eq=G['eq'],
            )
        ),
        'identity' : {"t": [1], "g": []},
        'inv'      : lambda bar: EG_to_BG(
            eg_inv(
                BG_to_EG(bar, G['identity']),
                inv=G['inv'], grplaw=G['grplaw'], identity=G['identity'], eq=G['eq'],
            )
        ),
        'eq'       : _eq_BG,
    }

# =============================================================================
# BG_tools
# =============================================================================
def _validate_bg_element(t, g, tol=1e-9):
    """
    Validate one BG element represented by:
        t = [t0, ..., tn]
        g = [g0, ..., g_{n-1}]
    where t_i >= 0 and sum t_i = 1.  Note len(t) = len(g) + 1.
    """
    if len(t) != len(g) + 1:
        raise ValueError(f"len(t)={len(t)} must equal len(g)+1={len(g)+1}")

    if len(t) == 0:
        raise ValueError("t must be nonempty (at least one element)")

    t = np.array(t, dtype=float)

    if np.any(t < -tol):
        raise ValueError("All t_i must be nonnegative")

    if abs(np.sum(t) - 1.0) > tol:
        raise ValueError(f"Sum of t_i must be 1, got {np.sum(t)}")

    return t, list(g)


def BG_to_EG(bar, G_identity):
    """
    Convert a BG bar element to an EG bar element by appending G's identity
    to the g-list. The t-list is unchanged.

    Parameters
    ----------
    bar : dict
        A BG bar element {"t": [...], "g": [...]} with len(t) = len(g) + 1.
    G_identity : object
        The identity element of the underlying group G.

    Returns
    -------
    dict
        An EG bar element {"t": [...], "g": [...]} with len(t) = len(g).
    """
    return {"t": list(bar["t"]),
            "g": list(bar["g"]) + [G_identity]}


def EG_to_BG(bar):
    """
    Convert an EG bar element to a BG bar element by dropping the last
    g-entry. The t-list is unchanged.

    Parameters
    ----------
    bar : dict
        An EG bar element {"t": [...], "g": [...]} with len(t) = len(g).

    Returns
    -------
    dict
        A BG bar element {"t": [...], "g": [...]} with len(t) = len(g) + 1.
    """
    return {"t": list(bar["t"]),
            "g": list(bar["g"])[:-1]}


# NOTE: BZ_to_S1 and MilEG_S1_to_CPinf were moved to representations.py
# (they are representation / visualization maps, not group-theoretic primitives).


# =============================================================================
# EG_tools
# Functions for working with EG bar elements (thin realization of EG).
# An EG bar element is represented as a dict {"t": [...], "g": [...]} where
#   t = [t0, ..., tn] with ti >= 0 and sum(ti) = 1
#   g = [g0, ..., gn] are group elements in G
#
# Functions:
#   _validate_eg_element : (internal) validate a (t, g) pair
#   _validate_bar        : (internal) validate a bar dict
#   plot_eg_bars         : visualize one or more EG elements as horizontal bars on [0,1]
#   is_reduced_bar       : check if a bar element is already in reduced form
#   reduced_form         : fully reduce a bar element
#   reduced_or_true      : return True if already reduced, else return the reduced form
#   eg_group_law         : compute the group product of two bar elements in EG
#   eg_inv               : compute the group inverse of a bar element in EG
#   homotopy_eg          : construct the homotopy contraction of a bar element
# =============================================================================

def _validate_eg_element(t, g, tol=1e-9):
    """
    Validate one EG element represented by:
        t = [t0, ..., tn]
        g = [g0, ..., gn]
    where t_i >= 0 and sum t_i = 1.
    """
    if len(t) != len(g):
        raise ValueError(f"len(t)={len(t)} must equal len(g)={len(g)}")

    if len(t) == 0:
        raise ValueError("t and g must be nonempty")

    t = np.array(t, dtype=float)

    if np.any(t < -tol):
        raise ValueError("All t_i must be nonnegative")

    if abs(np.sum(t) - 1.0) > tol:
        raise ValueError(f"Sum of t_i must be 1, got {np.sum(t)}")

    return t, list(g)

def _looks_like_single_bar(item):
    """
    True iff `item` directly represents ONE EG bar (dict form, or (t, g)
    tuple form) rather than a row (list) of bars.

    Disambiguation relies on the fact that t-lists always hold numbers: for
    tuple form (t, g), item[0] is the t-list and item[0][0] must be a Number.
    A row of bars, even a row of exactly 2 tuple-form bars, has item[0] equal
    to a whole bar (a dict, or a (t, g) pair) rather than a bare number.
    """
    if isinstance(item, dict):
        return True
    if isinstance(item, (list, tuple)) and len(item) == 2:
        t_cand = item[0]
        return (
            isinstance(t_cand, (list, tuple, np.ndarray))
            and len(t_cand) > 0
            and isinstance(t_cand[0], Number)
        )
    return False


def plot_eg_bars(
    elements,
    figsize_per_row=(10, 1.8),
    marker_size=120,
    g_label_offset=0.13,
    t_label_offset=0.11,
    title=None,
    show_t_values=True,
    bar_lw=3,
    boundary_lw=1.2,
    fontsize=11,
    position_title=True,
    end_labels=True,
):
    """
    Plot EG elements as horizontal bars on [0,1], arranged in a grid.

    `elements` can be either:
      - a flat list of bars [bar0, bar1, ...]
            -> one column, one bar per row (backward compatible)
      - a nested list of rows [[bar00, bar01, ...], [bar10, bar11, ...], ...]
            -> a matrix layout: elements[i][j] becomes the subplot at
               grid position (row i, column j), same convention as indexing
               a numpy array / nested list.
        Rows may have different lengths; the grid width is the longest row,
        and shorter rows leave their remaining cells blank.

    Each EG element is represented by:
        t = [t0, ..., tn]
        g = [g0, ..., gn]
    given either as a dict {'t': t, 'g': g} or as a (t, g) tuple.

    Display rules (per bar):
    - bar is [0,1]
    - t_i is shown at the midpoint of the i-th interval
    - green 'x' marker is shown at the RIGHT endpoint of the i-th interval
    - g_i is shown below that RIGHT endpoint
    """
    if not isinstance(elements, (list, tuple)) or len(elements) == 0:
        raise ValueError("elements must be a nonempty list")

    if _looks_like_single_bar(elements[0]):
        rows = [[e] for e in elements]      # flat input -> one bar per row
    else:
        rows = [list(r) for r in elements]  # already a list of rows

    def _parse_bar(elem):
        if isinstance(elem, dict):
            t, g = elem["t"], elem["g"]
        elif isinstance(elem, (list, tuple)) and len(elem) == 2:
            t, g = elem
        else:
            raise ValueError(
                "Each bar must be either {'t': [...], 'g': [...]} or ([...], [...])"
            )
        return _validate_eg_element(t, g)

    parsed_rows = [[_parse_bar(e) for e in row] for row in rows]

    nrows = len(parsed_rows)
    ncols = max(len(row) for row in parsed_rows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_row[0] * ncols, figsize_per_row[1] * nrows),
        squeeze=False
    )

    for r, row in enumerate(parsed_rows):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(row):
                ax.axis("off")
                continue

            t, g = row[c]
            endpoints = np.concatenate(([0.0], np.cumsum(t)))   # x_0=0, x_{i+1}=sum_{j<=i} t_j
            midpoints = 0.5 * (endpoints[:-1] + endpoints[1:])  # interval midpoints
            right_endpoints = endpoints[1:]                      # right endpoint of each interval

            # Main bar
            ax.plot([0, 1], [0, 0], color="black", lw=bar_lw, solid_capstyle="butt")

            # Green x markers at right endpoints of intervals
            ax.scatter(
                right_endpoints,
                np.zeros_like(right_endpoints),
                marker="x",
                s=marker_size,
                color="green",
                zorder=3
            )

            # g_i labels below right endpoints
            for x, label in zip(right_endpoints, g):
                ax.text(
                    x,
                    -g_label_offset,
                    str(label),
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    color='green'
                )

            # t_i labels at interval midpoints
            if show_t_values:
                for x, ti in zip(midpoints, t):
                    ax.text(
                        x,
                        t_label_offset,
                        f"{ti:.3g}",
                        ha="center",
                        va="bottom",
                        fontsize=fontsize - 1,
                        color="blue"
                    )

            # End labels 0 and 1
            if end_labels:
                ax.text(0, 0.09, "0", ha="center", va="bottom", fontsize=fontsize)
                ax.text(1, 0.09, "1", ha="center", va="bottom", fontsize=fontsize)
            # Ticks on 0 and 1
            ax.plot([0, 0], [-0.05, 0.05], color="black", lw=boundary_lw)
            ax.plot([1, 1], [-0.05, 0.05], color="black", lw=boundary_lw)

            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.28, 0.22)
            ax.axis("off")
            if position_title:
                ax.set_title(f"EG element ({r},{c})", fontsize=fontsize + 1, pad=8)

    if title is not None:
        fig.suptitle(title, fontsize=fontsize + 3, y=1.02)

    plt.tight_layout()
    plt.show()


def is_reduced_bar(bar, grplaw=Z['grplaw'], identity=Z['identity'], eq=_eq, tol=1e-12):
    """
    Return True iff the bar element is already reduced under the user's rules.

    The `eq` parameter is the group-aware equality used to compare g entries
    with `identity`. It defaults to `_eq` (tolerance-aware for numbers, strict
    `==` otherwise). For nested cases (e.g. EG over BG), pass the underlying
    group's own eq so that floating-point differences in nested bars are
    properly treated as equality.
    """
    t = list(bar["t"])
    g = list(bar["g"])

    if len(t) != len(g):
        raise ValueError("len(t) must equal len(g).")

    n = len(t)
    if n == 0:
        return True

    # Rule 1 and 2: any t_i = 0 makes it reducible (t entries are numeric)
    for ti in t:
        if _eq(ti, 0, tol=tol):
            return False

    # Rule 3: any g_i = identity for i < n-1 makes it reducible
    #         (use group-aware eq so nested-bar tolerance works correctly)
    for i in range(len(g) - 1):
        if eq(g[i], identity, tol=tol):
            return False

    return True


def reduced_form(bar, grplaw=Z['grplaw'], identity=Z['identity'], eq=_eq, tol=1e-12):
    """
    Fully reduce a bar element according to the thin realization rules.

    Input format:
        {
            "t": [...],
            "g": [...]
        }

    The `eq` parameter is the group-aware equality used to compare g entries
    with `identity` (Rule 3). Defaults to `_eq` (tolerance-aware for numbers,
    strict `==` otherwise). For nested cases (e.g. EG over BG), pass the
    underlying group's own eq.

    Returns a reduced bar element in the same format.
    """
    t = list(deepcopy(bar["t"]))
    g = list(deepcopy(bar["g"]))

    if len(t) != len(g):
        raise ValueError("len(t) must equal len(g).")

    changed = True
    while changed:
        changed = False

        # ---------------------------
        # Step A: remove zero t_i (t entries are numeric — _eq is correct)
        # ---------------------------
        i = 0
        while i < len(t):
            if _eq(t[i], 0, tol=tol):
                changed = True

                if i == 0:
                    # Rule 1: delete t0 and g0
                    del t[0]
                    del g[0]
                else:
                    # Rule 2: g_{i-1} <- grplaw(g_{i-1}, g_i), then delete t_i, g_i
                    g[i - 1] = grplaw(g[i - 1], g[i])
                    del t[i]
                    del g[i]

                # restart full scan after each reduction
                break
            i += 1

        if changed:
            continue

        # ---------------------------
        # Step B: remove identity g_i for i not last
        # (use group-aware eq so nested-bar tolerance works correctly)
        # ---------------------------
        i = 0
        while i < len(g) - 1:   # last g cannot be reduced
            if eq(g[i], identity, tol=tol):
                changed = True

                # Rule 3: merge t_i and t_{i+1}, delete g_i and t_{i+1}
                t[i] = t[i] + t[i + 1]
                del t[i + 1]
                del g[i]

                # restart full scan after each reduction
                break
            i += 1

    return {"t": t, "g": g}


def reduced_or_true(bar, grplaw=Z['grplaw'], identity=Z['identity'], eq=_eq, tol=1e-12):
    """
    If already reduced, return True.
    Otherwise return its reduced form.
    """
    if is_reduced_bar(bar, grplaw=grplaw, identity=identity, eq=eq, tol=tol):
        return True
    return reduced_form(bar, grplaw=grplaw, identity=identity, eq=eq, tol=tol)


def _validate_bar(bar, tol=1e-12):
    t = list(deepcopy(bar["t"]))
    g = list(deepcopy(bar["g"]))

    if len(t) != len(g):
        raise ValueError("len(t) must equal len(g).")
    if len(t) == 0:
        raise ValueError("bar element must be nonempty.")
    if any((isinstance(x, Number) and x < -tol) for x in t):
        raise ValueError("All t_i must be nonnegative.")
    if abs(sum(t) - 1.0) > tol:
        raise ValueError(f"Sum of t_i must be 1, got {sum(t)}.")

    return t, g


def eg_group_law(bar1, bar2, grplaw=Z['grplaw'], identity=Z['identity'], eq=_eq, tol=1e-12):
    """
    Group law on EG.

    Given two bar elements:
        (t_0,...,t_n),(g_0,...,g_n)
        (s_0,...,s_m),(h_0,...,h_m)

    Their vertex positions are cumulative sums:
        x_i = t_0 + ... + t_i
        y_j = s_0 + ... + s_j

    At each vertex position:
      - if both bars have a vertex there, combine labels via grplaw
      - if only one has a vertex there, copy that label

    Then reconstruct the new bar from the merged vertex positions,
    and finally reduce it.
    """
    t1, g1 = _validate_bar(bar1, tol=tol)
    t2, g2 = _validate_bar(bar2, tol=tol)

    # Cumulative vertex positions
    pos1 = list(deepcopy(t1))
    for i in range(1, len(pos1)):
        pos1[i] += pos1[i - 1]

    pos2 = list(deepcopy(t2))
    for i in range(1, len(pos2)):
        pos2[i] += pos2[i - 1]

    # Merge the two sorted position lists with tolerance,
    # while combining labels when positions coincide.
    i = j = 0
    merged_positions = []
    merged_labels = []

    while i < len(pos1) or j < len(pos2):
        if i < len(pos1) and j < len(pos2):
            if abs(pos1[i] - pos2[j]) <= tol:
                # Same position in both bars
                merged_positions.append((pos1[i] + pos2[j]) / 2.0)
                merged_labels.append(grplaw(g1[i], g2[j]))
                i += 1
                j += 1
            elif pos1[i] < pos2[j] - tol:
                merged_positions.append(pos1[i])
                merged_labels.append(g1[i])
                i += 1
            else:
                merged_positions.append(pos2[j])
                merged_labels.append(g2[j])
                j += 1
        elif i < len(pos1):
            merged_positions.append(pos1[i])
            merged_labels.append(g1[i])
            i += 1
        else:
            merged_positions.append(pos2[j])
            merged_labels.append(g2[j])
            j += 1

    # Reconstruct t from merged positions
    merged_t = []
    prev = 0.0
    for x in merged_positions:
        merged_t.append(x - prev)
        prev = x

    result = {"t": merged_t, "g": merged_labels}

    # Final reduction (use group-aware eq for identity check on nested g)
    return reduced_form(result, grplaw=grplaw, identity=identity, eq=eq, tol=tol)


def eg_inv(bar, inv=Z['inv'], grplaw=Z['grplaw'], identity=Z['identity'], eq=_eq, tol=1e-12):
    """
    Inverse of an EG bar element.

    Input:
        bar = {
            "t": [t0, ..., tn],
            "g": [g0, ..., gn]
        }

    Output:
        {
            "t": [t0, ..., tn],
            "g": [inv(g0), ..., inv(gn)]
        }

    Then reduce the result before returning.
    """
    t, g = _validate_bar(bar, tol=tol)

    result = {
        "t": list(deepcopy(t)),
        "g": [inv(gi) for gi in g]
    }

    return reduced_form(result, grplaw=grplaw, identity=identity, eq=eq, tol=tol)


def homotopy_eg(bar, time, inv=Z['inv'], grplaw=Z['grplaw'], identity=Z['identity'], eq=_eq, tol=1e-12):
    """
    Given an EG bar element
        (t0,...,tn), (g0,...,gn)
    and time in [0,1], return
        (1-time, time*t0, ..., time*tn),
        ((g0 * ... * gn)^(-1), g0, ..., gn)
    then reduce the result.

    Parameters
    ----------
    bar : dict
        {"t": [...], "g": [...]}
    time : float
        Number in [0,1].
    inv : callable
        Group inverse. Defaults to Z['inv'].
    grplaw : callable
        Group law. Defaults to Z['grplaw'].
    identity : object
        Identity element. Defaults to Z['identity'].
    tol : float
        Numerical tolerance.
    """
    if not isinstance(time, Number):
        raise ValueError("time must be a real number.")
    if time < -tol or time > 1 + tol:
        raise ValueError("time must lie in [0,1].")

    # clamp tiny numerical drift
    if abs(time) <= tol:
        time = 0.0
    elif abs(time - 1.0) <= tol:
        time = 1.0

    t, g = _validate_bar(bar, tol=tol)

    # Compute total group product g0 * g1 * ... * gn
    total = g[0]
    for gi in g[1:]:
        total = grplaw(total, gi)

    # New t-list
    new_t = [1.0 - time] + [time * ti for ti in t]

    # New g-list
    new_g = [inv(total)] + list(g)

    result = {"t": new_t, "g": new_g}

    # Final reduction (use group-aware eq for identity check on nested g)
    return reduced_form(result, grplaw=grplaw, identity=identity, eq=eq, tol=tol)
