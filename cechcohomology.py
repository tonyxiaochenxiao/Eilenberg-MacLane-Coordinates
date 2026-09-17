"""
cechcohomology.py
=================

Utilities for Čech cochains on the nerve of an open cover, valued in a group
and optionally restricted to a local section.

Design:
    - A Čech cochain is a `CechCochain` instance that carries its own
      `dim`, `cover`, `grp`, `func`, and `section`.
    - Calling `eta(simplex)` performs dimension / repeated-index checks
      via `CechCochain.__call__` and then delegates to `eta.func(simplex)`.
    - Tool functions (coboundary, lifted_eta, i_sharp, ...) accept a
      `CechCochain` and return a `CechCochain`. They never take cover/grp/
      section as separate arguments; those are read from the input's
      attributes.

All tool functions here are pure: they do not depend on any notebook-level
globals.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
import itertools
import warnings
import time

import numpy as np

import eg_tools


# =============================================================================
# Validation toggle
# =============================================================================
# Module-level switch. When False, the math-level validators below short-circuit
# and behave as no-ops (returning True without doing any work). Cheap structural
# checks in eg_tools (_validate_eg_element, _validate_bg_element, _validate_bar)
# are NOT affected — they remain on.
#
# Affected:
#   validate_cech_cochain           -> early `return True`
#   validate_cech_cocycle           -> early `return True`
#   validate_lifted_eta             -> early `return True`
#   validate_lifted_eta_2           -> early `return True`
#   validate_Q_lower_cech_on_F_EG   -> still computes & returns Q_eta;
#                                      only the δ(Q) == eta check is skipped.
#
# Toggle from a notebook with:   cechcohomology.VALIDATE = False
VALIDATE = True
import sheaves


@dataclass
class CechCochain:
    """
    A Čech cochain of dimension `dim`, defined on the nerve of an open `cover`,
    valued in a group `grp`, and restricted to a local `section` (a subset of
    data indices; None means the full dataset).

    Calling a CechCochain instance on a simplex returns a dict mapping data
    indices (in the simplex intersection, restricted to section) to group
    elements.

    Attributes
    ----------
    dim : int
        The dimension of the cochain (accepts (dim+1)-element simplices).
    cover : numpy array of dtype=object
        The open cover; cover[j] is the array of data indices in U_j.
    grp : dict
        The group the cochain is valued in (see eg_tools for group dicts).
    func : callable
        A function that takes a simplex (tuple of distinct ints of length
        dim+1) and returns a dict {idx: group element}.
    section : array-like of ints or None
        The local section V: the cochain is valued on U_{i0...idim} ∩ V.
        None means no restriction (V = full dataset).
    """
    dim     : int
    cover   : Any
    grp     : dict
    func    : Callable
    section : Optional[Any] = None

    def __call__(self, simplex):
        simplex = tuple(simplex)

        # Check for repeated indices
        if len(set(simplex)) < len(simplex):
            raise ValueError(f"Invalid simplex {simplex}: indices must be distinct.")

        # Dimension check
        if len(simplex) != self.dim + 1:
            raise ValueError(
                f"This cochain is a {self.dim}-cochain and expects "
                f"{self.dim + 1}-element simplices, but received a "
                f"{len(simplex)}-element simplex {simplex}."
            )

        return self.func(simplex)


# =============================================================================
# Work-in-progress validators (keep at top as reminders)
# =============================================================================

# TODO: not yet finalized.

# =============================================================================
# Table of contents (functions defined below). "TODO" marks items whose
# bodies are placeholders or known to be intentionally weaker than needed —
# they are flagged here so future revisions can find them quickly.
#
#   1.  Low-level utilities
#       - _permutation_sign
#       - compute_intersection
#
#   2.  Cover construction
#       - compute_open_cover
#
#   3.  Cochain validation and equality
#       - validate_cech_cochain                          (TODO: placeholder)
#       - _covers_equal                                  (TODO: placeholder)
#       - _sections_equal                                (TODO: placeholder)
#       - cech_cochains_equal                            (TODO: depends on above)
#
#   4.  Coboundary and cocycle validation
#       - compute_cech_coboundary
#       - validate_cech_cocycle
#
#   5.  Lifts and restrictions
#       - i_sharp
#       - j_sharp
#       - lifted_eta
#       - restrict_eta_to
#       - validate_lifted_eta
#
#   6.  Arithmetic on cochains
#       - add_cech_cochains
#       - inv_cech_cochains
#
#   7.  Gluing
#       - gluing_cech_cochains
#
#   8.  Sheaf section bridges
#       - zero_cocycle_to_sheaf_section
#       - sheaf_section_to_zero_cocycle
#
#   9.  Softness extensions
#       - extension_by_softness_of_F_EG
#       - extension_by_softness_of_cech_cochains         (TODO: g preconditions
#                                                         could be relaxed to
#                                                         per-simplex)
#       - extension_by_softness_of_zero_cocycles
#       - extension_by_softness_of_cech_cocycles
#
#  10.  Materialization
#       - _materialize
#
#  11.  Dimension lowering
#       - Q_lower_cech_dim_on_F_EG
#
#  12.  Additional validators
#       - validate_lifted_eta_2
#       - validate_Q_lower_cech_on_F_EG

# =============================================================================


# =============================================================================
# 1. Low-level utilities
# =============================================================================

def _permutation_sign(original):
    """
    Determine whether the permutation that sorts `original` into ascending
    order is odd or even.

    Returns True if the permutation is odd (negative sign, i.e. -1),
    and False if the permutation is even (positive sign, i.e. +1).

    Works for any tuple/list of distinct comparable elements, not just
    permutations of {0, 1, ..., N}. The parity is determined by the relative
    order (rank) of the elements, not their absolute values.

    Parameters
    ----------
    original : tuple or list of distinct comparable elements
        e.g. (1, 0, 2) or (0, 7, 3)

    Returns
    -------
    bool
        True if odd permutation (sign = -1), False if even permutation (sign = +1).
    """
    sorted_list = sorted(original)
    pos = {v: i for i, v in enumerate(sorted_list)}
    perm = [pos[v] for v in original]
    inversions = sum(
        1 for i in range(len(perm)) for j in range(i + 1, len(perm))
        if perm[i] > perm[j]
    )
    return inversions % 2 == 1


def compute_intersection(simplex, open_cover, section=None):
    """
    Compute the intersection of open sets indexed by a simplex, optionally
    further restricted to a local section V.

    Returns cover[i0] ∩ cover[i1] ∩ ... ∩ cover[in]  ∩  section  (if given).

    Parameters
    ----------
    simplex : tuple of ints (i0, i1, ..., in)
        Indices into the cover, representing a simplex in the nerve.
    open_cover : numpy array of dtype=object
        cover[j] is a numpy array of data indices in the j-th open set.
    section : array-like of ints, optional
        A local section V to further restrict the intersection. If None,
        no additional restriction is applied (equivalent to V = full dataset).

    Returns
    -------
    intersection : numpy array of ints
        Sorted array of data indices in cover[i0] ∩ ... ∩ cover[in] ∩ section.
        Empty array if the intersection is empty.
    """
    result = open_cover[simplex[0]]
    for i in simplex[1:]:
        result = np.intersect1d(result, open_cover[i])
    if section is not None:
        result = np.intersect1d(result, section)
    return result


# =============================================================================
# 2. Cover construction
# =============================================================================

def compute_open_cover(data, centers, radius, geodesic_distance):
    """
    Compute the open cover of data with respect to the given centers and radius.

    Parameters
    ----------
    data : array-like, shape (M, d)
        The dataset of points.
    centers : array-like, shape (N, d)
        The centers of the open balls.
    radius : float
        The radius of each open ball.
    geodesic_distance : function(x, y) -> float
        A function computing the distance between two points.

    Returns
    -------
    cover : numpy array of shape (N,), dtype=object
        cover[j] is a numpy array of indices i such that geodesic_distance(centers[j], data[i]) < radius.
    """
    cover = np.empty(len(centers), dtype=object)
    for j, center in enumerate(centers):
        cover[j] = np.array([i for i, x in enumerate(data) if geodesic_distance(center, x) < radius])
    return cover


# =============================================================================
# Coboundary and cocycle validation
# =============================================================================


# =============================================================================
# 3. Cochain validation and equality
# =============================================================================

# TODO: validate_cech_cochain currently runs a structural sanity
# check only; future revisions may add group-membership checks etc.
def validate_cech_cochain(eta):
    """
    Sanity check: verify that the CechCochain `eta` can be called without
    raising on every standard n-simplex, where n = eta.dim.

    Parameters
    ----------
    eta : CechCochain

    Returns
    -------
    True if every standard n-simplex is handled without exception.

    Raises
    ------
    ValueError
        If eta(simplex) raises on some standard n-simplex; the offending
        simplex and the underlying error are included in the message.

    IMPORTANT — what this function does NOT check:
        - It does NOT verify that the values eta(simplex)[i] are elements
          of any particular group G.
        - It does NOT check the Čech cocycle condition (delta eta = 0).
        - It does NOT check alternating consistency beyond what eta itself
          enforces.
    """
    if not VALIDATE:
        return True

    n = eta.dim
    N = len(eta.cover)

    for simplex in itertools.combinations(range(N), n + 1):
        try:
            eta(simplex)
        except Exception as e:
            raise ValueError(
                f"validate_cech_cochain failed on simplex {simplex}: {e}"
            ) from e

    return True


# TODO: placeholder helpers. Currently always return True (no check).
# Replace with real comparison logic for numpy object-arrays of numpy arrays
# (for covers) and "None or numpy array" (for sections).


# TODO: placeholder — always returns True. Replace with real
# comparison of two numpy object-arrays of index arrays.
def _covers_equal(cover1, cover2):
    """
    Check whether two open covers are equal.

    TODO: currently always returns True (no actual comparison).
    Real implementation should compare both the number of open sets and
    the contents (data indices) of each.
    """
    return True


# TODO: placeholder — always returns True. Replace with real
# comparison (handle None vs numpy array, element-wise compare).
def _sections_equal(section1, section2):
    """
    Check whether two sections are equal.

    TODO: currently always returns True (no actual comparison).
    Real implementation should handle the None case (full dataset) and
    compare numpy arrays elementwise.
    """
    return True


# ==============================================================
# Low-level utilities
# =============================================================================


# TODO: while the dim / grp checks and per-simplex value checks are
# real, cover- and section-equality currently fall through to the
# placeholders above. Will tighten once those are implemented.
def cech_cochains_equal(eta1, eta2, func=True, section=True):
    """
    Check whether two CechCochains eta1 and eta2 are compatible (and optionally
    equal in values).

    Always checks:
      - eta1.dim == eta2.dim
      - covers match (via _covers_equal)
      - groups match (via eta.grp['name'])

    Conditionally checks:
      - sections match (via _sections_equal), only when `section=True`.
      - values on every standard simplex agree, only when `func=True`.

    Passing `func=False` is useful as a compatibility check before combining
    two cochains that are structurally aligned but may differ in values.
    Passing `section=False` is useful when intentionally combining cochains
    over different sections (e.g. gluing two cochains whose sections are
    disjoint or merely overlap).

    Parameters
    ----------
    eta1, eta2 : CechCochain
    func : bool, default True
        Whether to compare the underlying functions (values on every standard
        simplex). If False, only the structural attributes are checked.
    section : bool, default True
        Whether to require eta1.section == eta2.section. If False, this
        check is skipped — the two cochains may live over different local
        sections.

    Returns
    -------
    True if no discrepancy is found.

    Raises
    ------
    ValueError
        With a detailed description of the first discrepancy found.
    """
    # --- Check dim ---
    if eta1.dim != eta2.dim:
        raise ValueError(
            f"Dimension mismatch: eta1 is a {eta1.dim}-cochain but "
            f"eta2 is a {eta2.dim}-cochain."
        )

    # --- Check cover ---
    if not _covers_equal(eta1.cover, eta2.cover):
        raise ValueError("Cover mismatch: eta1.cover and eta2.cover differ.")

    # --- Check group ---
    if eta1.grp['name'] != eta2.grp['name']:
        raise ValueError(
            f"Group mismatch: eta1 is valued in {eta1.grp['name']} "
            f"but eta2 is valued in {eta2.grp['name']}."
        )

    # --- Check section (only if section=True) ---
    if section:
        if not _sections_equal(eta1.section, eta2.section):
            raise ValueError("Section mismatch: eta1.section and eta2.section differ.")

    # --- Check keys (always) and values (only if func=True) on every
    #     standard simplex ---
    n = eta1.dim
    N = len(eta1.cover)

    for simplex in itertools.combinations(range(N), n + 1):
        d1 = eta1(simplex)
        d2 = eta2(simplex)

        keys1 = set(d1.keys())
        keys2 = set(d2.keys())

        if keys1 != keys2:
            only_in_1 = keys1 - keys2
            only_in_2 = keys2 - keys1
            msg = f"Key mismatch on simplex {simplex}:"
            if only_in_1:
                msg += f"\n  indices only in eta1: {sorted(only_in_1)}"
            if only_in_2:
                msg += f"\n  indices only in eta2: {sorted(only_in_2)}"
            raise ValueError(msg)

        if func:
            for idx in keys1:
                v1, v2 = d1[idx], d2[idx]
                if not eta1.grp['eq'](v1, v2):
                    raise ValueError(
                        f"Value mismatch on simplex {simplex}, index {idx}:\n"
                        f"  eta1 gives {v1}\n"
                        f"  eta2 gives {v2}"
                    )

    return True


# =============================================================================
# Cover construction
# =============================================================================


# =============================================================================
# 4. Coboundary and cocycle validation
# =============================================================================

def compute_cech_coboundary(eta):
    """
    Compute the Čech coboundary of an n-dimensional Čech cochain eta,
    returning a new (n+1)-dimensional CechCochain.

    eta is valued on intersections U_{i0...in} ∩ V -> G, where V = eta.section.
    The coboundary is then valued on U_{i0...in+1} ∩ V -> G (same section).

    The coboundary formula for a standard (n+1)-simplex σ = (i0, ..., i_{n+1}) is:
        (δη)(σ)[idx] = η(i1,...,i_{n+1})[idx]
                     · η(i0,i2,...,i_{n+1})[idx]^{-1}
                     · η(i0,i1,i3,...,i_{n+1})[idx]
                     · ...
    i.e. alternating grplaw and inv over the n+2 faces obtained by removing
    one element at a time.

    For non-standard input, the result is obtained by computing the standard
    case and applying grp['inv'] if the permutation is odd.

    Parameters
    ----------
    eta : CechCochain

    Returns
    -------
    CechCochain
        An (n+1)-dimensional cochain with the same cover, grp, and section
        as eta.
    """
    validate_cech_cochain(eta)

    n       = eta.dim
    cover   = eta.cover
    grp     = eta.grp
    section = eta.section

    def new_func(simplex):
        # dim / repeated-index checks already done by CechCochain.__call__
        sorted_simplex = tuple(sorted(simplex))

        # Non-standard input: recurse on sorted version, apply inv if odd permutation
        if simplex != sorted_simplex:
            is_odd = _permutation_sign(simplex)
            standard_result = new_func(sorted_simplex)
            if is_odd:
                return {idx: grp['inv'](v) for idx, v in standard_result.items()}
            else:
                return standard_result

        # Standard input: compute coboundary formula.
        # compute_intersection already restricts to section.
        indices = compute_intersection(sorted_simplex, cover, section)

        # Precompute eta on each face ONCE (independent of idx). Without this
        # hoist, eta(face) — which builds an entire dict — would be re-called
        # for every idx, giving an O(|indices|) blowup. For deeply nested
        # cochains this is the difference between milliseconds and minutes.
        face_values = []
        for k in range(n + 2):  # (n+1)-simplex has n+2 elements
            face = sorted_simplex[:k] + sorted_simplex[k + 1:]  # remove k-th element
            face_values.append(eta(face))

        result = {}
        for idx in indices:
            t = grp['identity']
            for k in range(n + 2):
                face_value = face_values[k][idx]
                if k % 2 == 0:
                    t = grp['grplaw'](t, face_value)
                else:
                    t = grp['grplaw'](t, grp['inv'](face_value))
            result[idx] = t

        return result

    return CechCochain(
        dim     = n + 1,
        cover   = cover,
        grp     = grp,
        func    = new_func,
        section = section,
    )


def validate_cech_cocycle(eta, tol=1e-10):
    """
    Validate whether a Čech cochain eta is a cocycle, i.e. whether its
    coboundary δη is identically the group identity at every index.

    Parameters
    ----------
    eta : CechCochain
    tol : float, default 1e-12
        Tolerance forwarded to eta.grp['eq'] when comparing computed values
        against the group identity. Loosen this (e.g. 1e-9) if floating-point
        accumulation in a long chain of group operations is making a
        bona-fide cocycle fail the default tolerance.

    Returns
    -------
    bool
        True if δη = identity (eta is a cocycle), False otherwise.

    Notes
    -----
    Equality with the identity is tested via eta.grp['eq'] to handle numerical
    tolerance (e.g. for S^1-valued cochains). For EG- and BG-valued cochains,
    the eq function is element-wise tolerance-aware (it forwards `tol` to the
    underlying group's eq).
    """
    if not VALIDATE:
        return True

    validate_cech_cochain(eta)
    delta_eta = compute_cech_coboundary(eta)

    N = len(eta.cover)
    for simplex in itertools.combinations(range(N), eta.dim + 2):
        values = delta_eta(simplex)
        for val in values.values():
            if not eta.grp['eq'](val, eta.grp['identity'], tol):
                print(eta.grp['identity'], " != ", val)
                return False

    return True


# =============================================================================
# Lifts and restrictions
# =============================================================================


# =============================================================================
# 5. Lifts and restrictions
# =============================================================================

def i_sharp(eta):
    """
    Lift a G-valued Čech cochain eta to an EG-valued Čech cochain,
    via the inclusion map i: G -> EG defined by i(g) = {"t": [1], "g": [g]}.

    For every simplex and every index, the value is lifted as:
        eta(simplex)[idx] = g   -->   new_eta(simplex)[idx] = {"t": [1], "g": [g]}

    The new cochain has the same dim, cover, and section as eta, but its
    group is replaced by EG = eg_tools.make_EG(eta.grp).

    Parameters
    ----------
    eta : CechCochain
        A G-valued Čech cochain.

    Returns
    -------
    CechCochain
        An EG-valued Čech cochain with the same dim, cover, and section.
    """
    G  = eta.grp
    EG = eg_tools.make_EG(G)

    def new_func(simplex):
        old_result = eta(simplex)
        return {idx: {"t": [1], "g": [v]} for idx, v in old_result.items()}

    return CechCochain(
        dim     = eta.dim,
        cover   = eta.cover,
        grp     = EG,
        func    = new_func,
        section = eta.section,
    )

def j_sharp(eta):
    """
    Push an EG-valued Čech cochain forward to a BG-valued Čech cochain
    via the quotient EG -> BG = EG/G.

    Concretely, on every simplex and every index, the bar element value
    in EG is mapped to BG by dropping its last g-entry (see
    eg_tools.EG_to_BG):

        eta(simplex)[idx] = {"t": [t0,..,tn], "g": [g0,..,gn]}  in EG
            -->
        new_eta(simplex)[idx] = {"t": [t0,..,tn], "g": [g0,..,g_{n-1}]}  in BG

    The new cochain has the same dim, cover, and section as eta, but its
    group is replaced by BG = eg_tools.make_BG(G) where G = eta.grp['source'].

    Parameters
    ----------
    eta : CechCochain
        An EG-valued Čech cochain. eta.grp must have been constructed via
        eg_tools.make_EG (so eta.grp['source'] is itself a group dict).

    Returns
    -------
    CechCochain
        A BG-valued Čech cochain with the same dim, cover, and section.

    Raises
    ------
    ValueError
        If eta.grp is not EG-valued (i.e. eta.grp['source'] is not a group dict).
    """
    source = eta.grp.get('source')
    if not isinstance(source, dict):
        raise ValueError(
            f"j_sharp requires eta to be EG-valued (eta.grp must be "
            f"constructed via eg_tools.make_EG). Got eta.grp with name "
            f"'{eta.grp.get('name', '?')}' and source {source!r}."
        )

    EG = eta.grp
    G  = source
    BG = eg_tools.make_BG(G)

    def new_func(simplex):
        old_result = eta(simplex)
        return {idx: eg_tools.EG_to_BG(v) for idx, v in old_result.items()}

    return CechCochain(
        dim     = eta.dim,
        cover   = eta.cover,
        grp     = BG,
        func    = new_func,
        section = eta.section,
    )


def lifted_eta(eta):
    """
    For each open set U_i, construct a (n-1)-dimensional CechCochain lifted_eta[i],
    where eta is an n-dimensional CechCochain.

    Each lifted_eta[i] has:
      - dim = eta.dim - 1
      - cover = eta.cover
      - grp = eta.grp
      - section = U_i ∩ eta.section  (if eta.section is None, this is just U_i)

    For a standard (n-1)-simplex (i0, ..., i_{n-1}):
      - If i is already in the simplex: return grp['identity'] for all indices
        in compute_intersection((i0,...,i_{n-1}), cover, lifted_section), where
        lifted_section = U_i ∩ eta.section.
      - If i is not in the simplex: return eta((i, i0,...,i_{n-1})), which is
        an n-simplex (possibly non-standard; eta handles the sign and already
        respects eta.section. Since i is now a factor, the result's keys lie
        in U_i ∩ ... ∩ eta.section, exactly the required local section).

    Non-standard input to lifted_eta[i] is handled by recursing on the sorted
    simplex and applying grp['inv'] if the permutation is odd.

    Parameters
    ----------
    eta : CechCochain

    Returns
    -------
    list of CechCochain
        A list of length N = len(cover). lifted_eta[i] is a (dim-1)-cochain
        with section = U_i ∩ eta.section.
    """
    validate_cech_cochain(eta)

    n     = eta.dim
    cover = eta.cover
    grp   = eta.grp
    N     = len(cover)

    def make_entry(i):
        # lifted section: U_i ∩ eta.section
        if eta.section is None:
            lifted_section = cover[i]
        else:
            lifted_section = np.intersect1d(cover[i], eta.section)

        def lifted_eta_i_func(simplex):
            # dim / repeated-index checks done by CechCochain.__call__
            sorted_simplex = tuple(sorted(simplex))

            # Non-standard input: recurse on sorted version, apply inv if odd
            if simplex != sorted_simplex:
                is_odd = _permutation_sign(simplex)
                standard_result = lifted_eta_i_func(sorted_simplex)
                if is_odd:
                    return {idx: grp['inv'](v) for idx, v in standard_result.items()}
                else:
                    return standard_result

            # Standard input, case 1: i already in simplex -> all identity,
            # restricted to lifted_section (= U_i ∩ eta.section).
            if i in sorted_simplex:
                indices = compute_intersection(sorted_simplex, cover, lifted_section)
                return {idx: grp['identity'] for idx in indices}

            # Standard input, case 2: i not in simplex -> call eta with i prepended.
            # (i,) + sorted_simplex may be non-standard; eta handles the sign.
            # eta already respects eta.section, and since i is a factor of the
            # resulting n-simplex, the keys lie in U_i ∩ ... ∩ eta.section,
            # which matches lifted_section on the restriction to this simplex.
            return eta((i,) + sorted_simplex)

        return CechCochain(
            dim     = n - 1,
            cover   = cover,
            grp     = grp,
            func    = lifted_eta_i_func,
            section = lifted_section,
        )

    return [make_entry(i) for i in range(N)]


def restrict_eta_to(eta, U):
    """
    Restrict a Čech cochain eta to an open set U, returning a new CechCochain.

    For each n-simplex, new_eta(simplex) is the same dict as eta(simplex)
    but with only those indices that belong to U retained.

    The new section is U ∩ eta.section (if eta.section is None, it's just U).

    Parameters
    ----------
    eta : CechCochain
    U : array-like of ints
        A collection of data indices representing the open set to restrict to.

    Returns
    -------
    CechCochain
        The restricted cochain with updated section.
    """
    U_arr = np.asarray(list(U))
    U_set = set(U_arr.tolist())

    def new_func(simplex):
        old_result = eta(simplex)
        return {idx: v for idx, v in old_result.items() if idx in U_set}

    # New section: U ∩ eta.section (if eta.section is None, new section is U)
    if eta.section is None:
        new_section = U_arr
    else:
        new_section = np.intersect1d(U_arr, eta.section)

    return CechCochain(
        dim     = eta.dim,
        cover   = eta.cover,
        grp     = eta.grp,
        func    = new_func,
        section = new_section,
    )


def validate_lifted_eta(eta, i=None):
    """
    Validate the lifted_eta construction by checking, for each i, that:

        δ(lifted_eta[i]) == restrict_eta_to(eta, cover[i])

    as CechCochains (they agree on every standard simplex, for every index in
    U_i, up to eta.grp['eq']).

    Since lifted_eta[i] has section=cover[i], its coboundary automatically has
    section=cover[i]. restrict_eta_to(eta, cover[i]) also has section=cover[i]
    (if eta.section is None) or cover[i] ∩ eta.section (otherwise). The two
    should agree on all simplices via cech_cochains_equal.

    Parameters
    ----------
    eta : CechCochain
    i : int or None, optional (default: None)
        If None, check the identity for every i in range(len(eta.cover)).
        If a specific int is given, only check that single i. Useful for
        pinpointing a failure or for validating incrementally.

    Returns
    -------
    True if the identity δ(lifted_eta[i]) = eta|_{U_i} holds for the
    requested i (or for every i if i is None).

    Raises
    ------
    ValueError
        With a detailed message indicating which i failed and why, or if
        the supplied i is out of range.
    """
    if not VALIDATE:
        return True

    N = len(eta.cover)

    if i is None:
        indices_to_check = range(N)
    else:
        if not (0 <= i < N):
            raise ValueError(
                f"validate_lifted_eta: i must be in [0, {N - 1}] "
                f"(or None for the full sweep), got i = {i}."
            )
        indices_to_check = [i]

    le = lifted_eta(eta)

    for k in indices_to_check:
        delta_le_k     = compute_cech_coboundary(le[k])
        eta_restricted = restrict_eta_to(eta, eta.cover[k])

        try:
            cech_cochains_equal(delta_le_k, eta_restricted)
        except ValueError as e:
            raise ValueError(
                f"validate_lifted_eta failed for i={k}:\n{e}"
            ) from e

    return True


# =============================================================================
# 6. Arithmetic on cochains
# =============================================================================

def add_cech_cochains(eta1, eta2):
    """
    Add two compatible Čech cochains pointwise in their common group.

    First verifies compatibility (dim, cover, grp, section, and keys on every
    simplex) via cech_cochains_equal(eta1, eta2, func=False). Values are
    intentionally NOT compared — otherwise compatible cochains with different
    values could not be added.

    The resulting cochain inherits dim, cover, grp, and section from eta1
    (equal to eta2's by the compatibility check). On each simplex, its func
    returns:
        {key: grplaw(eta1(simplex)[key], eta2(simplex)[key]) for key in ...}

    Parameters
    ----------
    eta1, eta2 : CechCochain

    Returns
    -------
    CechCochain
        The pointwise sum eta1 + eta2.
    """
    cech_cochains_equal(eta1, eta2, func=False)

    grplaw = eta1.grp['grplaw']

    def new_func(simplex):
        d1 = eta1(simplex)
        d2 = eta2(simplex)
        # keys of d1 and d2 agree (verified above)
        return {key: grplaw(d1[key], d2[key]) for key in d1}

    return CechCochain(
        dim     = eta1.dim,
        cover   = eta1.cover,
        grp     = eta1.grp,
        func    = new_func,
        section = eta1.section,
    )


def inv_cech_cochains(eta):
    """
    Invert a Čech cochain pointwise in its group.

    On each simplex, the new func returns:
        {key: inv(eta(simplex)[key]) for key in ...}

    The resulting cochain inherits dim, cover, grp, and section from eta.

    Parameters
    ----------
    eta : CechCochain

    Returns
    -------
    CechCochain
        The pointwise inverse -eta.
    """
    inv = eta.grp['inv']

    def new_func(simplex):
        d = eta(simplex)
        return {key: inv(v) for key, v in d.items()}

    return CechCochain(
        dim     = eta.dim,
        cover   = eta.cover,
        grp     = eta.grp,
        func    = new_func,
        section = eta.section,
    )


# =============================================================================
# 7. Gluing
# =============================================================================

def gluing_cech_cochains(eta1, eta2):
    """
    Glue two Čech cochains eta1 and eta2 over possibly overlapping sections
    into a single cochain over the union of their sections.

    Preconditions:
      - eta1 and eta2 must be structurally compatible (same dim, cover, grp).
      - Their values on the intersection of the two sections must agree
        (otherwise gluing is ill-defined).

    Section resolution:
      If eta.section is None, it is interpreted as the union of all U_i in
      eta.cover (the "effective" section).

    Verification:
      Let sec_k = resolved section of eta_k; intersection = sec1 ∩ sec2.
      Restrict both cochains to `intersection` and call
          cech_cochains_equal(eta1_restrict, eta2_restrict, func=True, section=True)
      which (under the current placeholders) checks dim, grp (by name), and
      values on every standard simplex. Raises ValueError on any mismatch.

      TODO: once _covers_equal and _sections_equal in cechcohomology.py are
      implemented (not placeholders), the cover and section equality will
      also be properly enforced here.

    Output:
      new_eta with:
        - dim, cover, grp inherited from eta1 (and equal to eta2's)
        - section = sec1 ∪ sec2
        - on each simplex, the dict is the key-wise union of eta1(simplex)
          and eta2(simplex). Overlap keys have consistent values (verified
          above), so updating d2 into d1 is safe.

    Non-standard simplex input is handled automatically: since eta1 and eta2
    each handle their own alternating sign, the union of their dicts is also
    alternating.

    Parameters
    ----------
    eta1, eta2 : CechCochain

    Returns
    -------
    CechCochain
        The glued cochain over sec1 ∪ sec2.

    Raises
    ------
    ValueError
        Via cech_cochains_equal if the two cochains are incompatible or
        disagree on the intersection of their sections.
    """
    # --- Resolve sections (treat None as ∪ cover) ---
    def _resolve_section(eta):
        if eta.section is not None:
            return np.asarray(eta.section)
        u = set()
        for arr in eta.cover:
            u.update(int(i) for i in arr)
        return np.array(sorted(u))

    sec1 = _resolve_section(eta1)
    sec2 = _resolve_section(eta2)

    intersection  = np.intersect1d(sec1, sec2)
    union_section = np.union1d(sec1, sec2)

    # --- Verify agreement on the intersection ---
    eta1_on_int = restrict_eta_to(eta1, intersection)
    eta2_on_int = restrict_eta_to(eta2, intersection)
    cech_cochains_equal(eta1_on_int, eta2_on_int, func=True, section=True)

    # --- Build the glued cochain ---
    def new_func(simplex):
        d1 = eta1(simplex)
        d2 = eta2(simplex)
        result = dict(d1)
        result.update(d2)   # overlap keys agree (verified), so update is safe
        return result

    return CechCochain(
        dim     = eta1.dim,
        cover   = eta1.cover,
        grp     = eta1.grp,
        func    = new_func,
        section = union_section,
    )


# =============================================================================
# 8. Sheaf section bridges
# =============================================================================

def zero_cocycle_to_sheaf_section(eta):
    """
    Realize the isomorphism ker d^0 ≅ F_G(V) as a sheaf map.

    Given a 0-dimensional Čech cocycle eta valued in some group G, return
    the corresponding element of F_G(domain) — i.e. a SheafSection whose
    values glue the per-U_i values of eta into a single well-defined
    function on the union.

    Because eta is a cocycle, for every data index idx present in multiple
    eta((i,))'s, the values agree; so the pointwise function
        idx  ->  eta((i,))[idx]   (for any i with idx in U_i ∩ section)
    is well-defined.

    The `domain` of the returned SheafSection is:
      - eta.section                        if eta.section is not None
      - ∪_i U_i  (= union of all cover sets)  if eta.section is None

    Implementation note: we precompute a single merged dict
    `{idx: value}` at construction time (one pass over all standard
    0-simplices). This gives O(1) lookup on each call, and also fails
    fast if some idx in `domain` is not covered by any U_i.

    Parameters
    ----------
    eta : CechCochain
        Must have eta.dim == 0 and must be a cocycle.

    Returns
    -------
    sheaves.SheafSection
        An element of F_G(domain), where G = eta.grp.

    Raises
    ------
    ValueError
        - if eta.dim != 0,
        - if eta is not a cocycle,
        - if some idx in `domain` is not present in any eta((i,)).
    """
    # --- 1. Dimension check ---
    if eta.dim != 0:
        raise ValueError(
            f"zero_cocycle_to_sheaf_section requires eta.dim == 0, "
            f"but received a {eta.dim}-cochain."
        )

    # --- 2. Cocycle check ---
    if not validate_cech_cocycle(eta):
        raise ValueError(
            "zero_cocycle_to_sheaf_section requires eta to be a cocycle, "
            "but validate_cech_cocycle(eta) returned False."
        )

    # --- 3. Precompute merged dict by walking over standard 0-simplices ---
    #     Because eta is a cocycle, repeated keys across i get consistent
    #     values; dict.update simply overwrites with the same value.
    N = len(eta.cover)
    merged = {}
    for i in range(N):
        for idx, val in eta((i,)).items():
            merged[int(idx)] = val

    # --- 4. Determine domain ---
    if eta.section is None:
        # Union of all U_i (= exactly the keys of `merged`, since merged
        # picks up every index that appears in any eta((i,))).
        domain = np.array(sorted(merged.keys()))
    else:
        domain = eta.section

    domain_set = set(int(i) for i in domain)

    # --- 5. Fail-fast: every idx in domain must be in merged ---
    missing = domain_set - set(merged.keys())
    if missing:
        missing_sorted = sorted(missing)
        preview = missing_sorted[:5]
        more = "" if len(missing_sorted) <= 5 else f" (+{len(missing_sorted) - 5} more)"
        raise ValueError(
            f"zero_cocycle_to_sheaf_section: {len(missing)} index/indices "
            f"in domain are not covered by any U_i, so their value under "
            f"the glued section is undefined. "
            f"First few missing: {preview}{more}."
        )

    # --- 6. Build the SheafSection ---
    def sheaf_func(idx):
        return merged[int(idx)]

    return sheaves.SheafSection(
        grp    = eta.grp,
        func   = sheaf_func,
        domain = domain,
    )


def sheaf_section_to_zero_cocycle(s, cover):
    """
    Inverse direction of the isomorphism ker d^0 ≅ F_G(V).

    Given a SheafSection s ∈ F_G(U) and an open cover, build a 0-dimensional
    Čech cochain eta with:
      - eta.dim     = 0
      - eta.cover   = cover
      - eta.grp     = s.grp
      - eta.section = s.domain
    and whose func on a standard 0-simplex (i,) returns
        { idx : s(idx)  for idx in cover[i] ∩ s.domain }.

    (By construction this is a cocycle: for any standard 1-simplex (i, j),
    (δη)((i, j))[idx] = η((j,))[idx] - η((i,))[idx] = s(idx) - s(idx) = 0
    for every idx in U_i ∩ U_j ∩ s.domain.)

    Parameters
    ----------
    s : sheaves.SheafSection
        An element of F_G(U), where U = s.domain.
    cover : numpy array of dtype=object
        The open cover to use for the resulting cochain.

    Returns
    -------
    CechCochain
        A 0-cochain whose value on (i,) is s restricted to cover[i] ∩ s.domain.
    """
    domain_set = set(int(i) for i in s.domain)

    def new_func(simplex):
        # simplex is guaranteed to be a 1-tuple by CechCochain.__call__
        i = simplex[0]
        result = {}
        for idx in cover[i]:
            idx_int = int(idx)
            if idx_int in domain_set:
                result[idx_int] = s(idx_int)
        return result

    return CechCochain(
        dim     = 0,
        cover   = cover,
        grp     = s.grp,
        func    = new_func,
        section = s.domain,
    )


# =============================================================================
# 9. Softness extensions
# =============================================================================

def extension_by_softness_of_F_EG(s, V, g, X, tol=1e-9):
    """
    Extend an EG-valued sheaf section s ∈ F_{EG}(U) to a section
    s_tilde ∈ F_{EG}(X) by "softening" with a [0,1]-valued function g.

    Preconditions (verified; raises ValueError if violated):
        - s.grp must be EG-valued, i.e. s.grp['source'] must itself be a
          group dict (EG is produced by eg_tools.make_EG).
        - g(v) == 1 for every v in V          (up to tol)
        - g(x) == 0 for every x in X \\ U     (up to tol)

    where U = s.domain.

    Construction:
        s_tilde(x) =
            homotopy_eg(s(x), g(x), inv=G.inv, grplaw=G.grplaw, identity=G.identity),
                                                                 if x ∈ U,
            EG.identity,                                         if x ∈ X \\ U.
    Here G = s.grp['source'] is the "source" group underlying EG = s.grp.

    Parameters
    ----------
    s : sheaves.SheafSection
        An EG-valued section on U = s.domain. s.grp must be an EG group
        (constructed via eg_tools.make_EG) so that s.grp['source'] gives
        the underlying group G.
    V : array-like of ints
        A "core" index set, expected to be a subset of U, on which g ≡ 1.
    g : callable(idx) -> float in [0,1]
        A partition-of-unity-like function defined on (at least) X.
    X : array-like of ints
        The target domain of the extension. Expected to be a superset of U.
    tol : float, default 1e-9
        Tolerance for the g == 1 (on V) and g == 0 (on X \\ U) checks.

    Returns
    -------
    sheaves.SheafSection
        s_tilde, an EG-valued section on X.
    """
    # --- Require s to be EG-valued ---
    source = s.grp.get('source')
    if not isinstance(source, dict):
        raise ValueError(
            f"extension_by_softness_of_F_EG requires s to be EG-valued "
            f"(s.grp must be constructed via eg_tools.make_EG). "
            f"Got s.grp with name '{s.grp.get('name', '?')}' and source "
            f"{source!r}. A general group G does not admit this 'softness' "
            f"extension — only EG(G), whose elements can be contracted to "
            f"the identity via homotopy_eg."
        )

    EG = s.grp
    G  = source  # = EG['source']

    U_set = set(int(i) for i in s.domain)
    V_set = set(int(i) for i in V)
    X_set = set(int(i) for i in X)

    # --- Precondition 1: g == 1 on V ---
    for v in V_set:
        val = g(v)
        if abs(val - 1.0) > tol:
            raise ValueError(
                f"extension_by_softness_of_F_EG: g({v}) = {val}, "
                f"expected 1 on V (within tol={tol})."
            )

    # --- Precondition 2: g == 0 on X \ U ---
    for x in X_set - U_set:
        val = g(x)
        if abs(val) > tol:
            raise ValueError(
                f"extension_by_softness_of_F_EG: g({x}) = {val}, "
                f"expected 0 on X \\ U (within tol={tol})."
            )

    # --- Build s_tilde ---
    EG_identity = EG['identity']

    def s_tilde_func(idx):
        if int(idx) in U_set:
            return eg_tools.homotopy_eg(
                s(idx), g(idx),
                inv      = G['inv'],
                grplaw   = G['grplaw'],
                identity = G['identity'],
                eq       = G['eq'],
            )
        else:
            return EG_identity

    return sheaves.SheafSection(
        grp    = EG,
        func   = s_tilde_func,
        domain = X,
    )


# TODO: the g ≡ 1 on V and g ≡ 0 on X∖U preconditions are checked
# globally here; mathematically only their restriction to V ∩ ∩σ and
# (X∖U) ∩ ∩σ per simplex is needed. See the NOTE TO THE AUTHOR in the
# docstring.
def extension_by_softness_of_cech_cochains(eta, V, g, X, tol=1e-9):
    """
    Extend an EG-valued n-dimensional Čech cochain from section=U (= eta.section)
    to section=X by "softening" with a [0,1]-valued function g.

    Mirrors `extension_by_softness_of_F_EG` but at the level of cochains of
    arbitrary dimension. The returned new_eta has:
        - new_eta.dim     = eta.dim
        - new_eta.cover   = eta.cover
        - new_eta.grp     = eta.grp               (still EG)
        - new_eta.section = X

    For a standard n-simplex σ, new_eta(σ)[idx] is defined as:
        homotopy_eg(eta(σ)[idx], g(idx), inv=G.inv, grplaw=G.grplaw, identity=G.identity),
                                                       if idx ∈ U ∩ ∩σ,
        EG.identity,                                   if idx ∈ (X ∖ U) ∩ ∩σ.

    Here G = eta.grp['source'].

    NOTE TO THE AUTHOR (stronger-than-necessary precondition):
        This implementation checks that g ≡ 1 on the *entire* V and g ≡ 0 on
        the *entire* X ∖ U — a global condition. Mathematically, for each
        simplex σ we only need g ≡ 1 on V ∩ ∩σ and g ≡ 0 on (X ∖ U) ∩ ∩σ.
        The current (global) check is therefore stronger than necessary.
        If a future construction benefits from a simplex-dependent g (i.e.
        a different g per simplex σ that only needs to satisfy the conditions
        on its own intersection), this check can be relaxed accordingly.

    Preconditions (raises ValueError if violated):
      - eta.grp must be EG-valued (eta.grp['source'] is a group dict).
      - g ≡ 1 on V  (within tol)
      - g ≡ 0 on X ∖ U  (within tol)

    Parameters
    ----------
    eta : CechCochain
        An EG-valued n-cochain with eta.section = U (or None meaning ∪ cover[i]).
    V : array-like of ints
        "Core" index set, expected subset of U, on which g ≡ 1.
    g : callable(idx) -> float in [0,1]
        The softening function.
    X : array-like of ints
        Target section (expected superset of U).
    tol : float, default 1e-9
        Tolerance for the g == 1 and g == 0 preconditions.

    Returns
    -------
    CechCochain
        new_eta, an EG-valued n-cochain with new_eta.section = X.
    """
    # --- Require eta to be EG-valued ---
    source = eta.grp.get('source')
    if not isinstance(source, dict):
        raise ValueError(
            f"extension_by_softness_of_cech_cochains requires eta to be "
            f"EG-valued (eta.grp must be constructed via eg_tools.make_EG). "
            f"Got eta.grp with name '{eta.grp.get('name', '?')}' and source "
            f"{source!r}. A general group G does not admit this 'softness' "
            f"extension — only EG(G), whose elements can be contracted to "
            f"the identity via homotopy_eg."
        )

    EG          = eta.grp
    G           = source  # = EG['source']
    EG_identity = EG['identity']
    cover       = eta.cover

    # Build U_set. If eta.section is None (i.e. "the whole ambient space"),
    # interpret U as the union of all cover sets, which is the only portion
    # of the ambient space that eta could ever assign a value to.
    if eta.section is None:
        U_set = set()
        for arr in cover:
            U_set.update(int(i) for i in arr)
    else:
        U_set = set(int(i) for i in eta.section)

    V_set = set(int(i) for i in V)
    X_set = set(int(i) for i in X)
    X_arr = np.asarray(X)

    # --- Precondition 1: g == 1 on V ---
    for v in V_set:
        val = g(v)
        if abs(val - 1.0) > tol:
            raise ValueError(
                f"extension_by_softness_of_cech_cochains: g({v}) = {val}, "
                f"expected 1 on V (within tol={tol})."
            )

    # --- Precondition 2: g == 0 on X \ U ---
    for x in X_set - U_set:
        val = g(x)
        if abs(val) > tol:
            raise ValueError(
                f"extension_by_softness_of_cech_cochains: g({x}) = {val}, "
                f"expected 0 on X \\ U (within tol={tol})."
            )

    # --- Build new_eta ---
    def new_func(simplex):
        # dim / repeated-index checks already done by CechCochain.__call__
        sorted_simplex = tuple(sorted(simplex))

        # Non-standard input: recurse on sorted, apply EG inv if odd permutation
        if simplex != sorted_simplex:
            is_odd = _permutation_sign(simplex)
            standard_result = new_func(sorted_simplex)
            if is_odd:
                return {idx: EG['inv'](v) for idx, v in standard_result.items()}
            else:
                return standard_result

        # Standard input
        eta_result = eta(sorted_simplex)  # dict on ∩sorted_simplex ∩ U
        indices_X  = compute_intersection(sorted_simplex, cover, section=X_arr)

        result = {}
        for idx in indices_X:
            idx_int = int(idx)
            if idx_int in U_set:
                # idx ∈ ∩σ ∩ U : apply homotopy
                result[idx_int] = eg_tools.homotopy_eg(
                    eta_result[idx_int], g(idx_int),
                    inv      = G['inv'],
                    grplaw   = G['grplaw'],
                    identity = G['identity'],
                    eq       = G['eq'],
                )
            else:
                # idx ∈ ∩σ ∩ (X \ U) : EG identity
                result[idx_int] = EG_identity
        return result

    return CechCochain(
        dim     = eta.dim,
        cover   = cover,
        grp     = eta.grp,
        func    = new_func,
        section = X,
    )


def extension_by_softness_of_zero_cocycles(eta, V, g, X, tol=1e-9, new_cover=None):
    """
    Extend a 0-dimensional Čech cocycle from section=U (= eta.section) to
    section=X, keeping the values on V unchanged.

    This is the Čech-cocycle counterpart of `extension_by_softness_of_F_EG`.
    The construction is:

        eta  -->  s = zero_cocycle_to_sheaf_section(eta)     ∈ F_EG(U)
            -->  s_tilde = extension_by_softness_of_F_EG(
                               s, V, g, X, tol=tol)           ∈ F_EG(X)
            -->  new_eta = sheaf_section_to_zero_cocycle(
                               s_tilde, new_cover)           0-cochain on X

    Preconditions:
      - eta.grp must be an EG group (constructed via eg_tools.make_EG), i.e.
        eta.grp['source'] must itself be a group dict. The "softness" step
        uses eg_tools.homotopy_eg, which is only defined on EG bar elements;
        a general group G has no homotopy to the identity, so this extension
        is not well-defined for non-EG cochains.
      - eta.dim == 0 and eta is a cocycle (enforced by zero_cocycle_to_sheaf_section)
      - g ≡ 1 on V, g ≡ 0 on X ∖ U  (enforced by extension_by_softness_of_F_EG)

    Parameters
    ----------
    eta : CechCochain
        A 0-dimensional EG-valued cocycle with eta.section = U.
    V : array-like of ints
        A "core" index set, expected subset of U, on which g ≡ 1.
    g : callable(idx) -> float in [0,1]
        The softening function.
    X : array-like of ints
        The target section (expected superset of U).
    tol : float, default 1e-9
        Tolerance for the g == 1 and g == 0 preconditions.
    new_cover : numpy array of dtype=object, optional
        The open cover to attach to the returned cochain. If None (default),
        `eta.cover` is reused. In general you may want a different cover for
        the extended cochain (e.g. one associated with a larger radius).

    Returns
    -------
    CechCochain
        new_eta: a 0-cochain with new_eta.section = X, agreeing with eta on V.
    """
    # --- Require eta to be EG-valued ---
    source = eta.grp.get('source')
    if not isinstance(source, dict):
        raise ValueError(
            f"extension_by_softness_of_zero_cocycles requires eta to be "
            f"EG-valued (eta.grp must be constructed via eg_tools.make_EG). "
            f"Got eta.grp with name '{eta.grp.get('name', '?')}' and "
            f"source {source!r}. A general group G does not admit this "
            f"'softness' extension — only EG(G), whose elements can be "
            f"contracted to the identity via homotopy_eg."
        )

    if new_cover is None:
        new_cover = eta.cover

    s       = zero_cocycle_to_sheaf_section(eta)
    s_tilde = extension_by_softness_of_F_EG(s, V, g, X, tol=tol)
    new_eta = sheaf_section_to_zero_cocycle(s_tilde, new_cover)
    return new_eta


def extension_by_softness_of_cech_cocycles(eta, V, g, X, i=None, tol=1e-9, new_cover=None):
    """
    Extend an EG-valued Čech cocycle from section=eta.section to section=X,
    uniformly across all dimensions.
    注意：这只是一个局部的extension函数，从U_i的某个子集extend到U_i。我们全局的extension函数目前不需要做，因为我们的输入domain永远是某个U_i的一个子集。
    
    Dispatch on `eta.dim`:
      - eta.dim == 0 : delegate to extension_by_softness_of_zero_cocycles(
                         eta, V, g, X, tol=tol, new_cover=new_cover).
                       In this case `i` is ignored (not needed).
      - eta.dim >= 1 : use the "lifted preimage" construction, which requires
                       a specific index `i` such that eta.section ⊆ cover[i].
                       In this case `new_cover` has no effect (the output
                       cover is forced to be eta.cover by the coboundary
                       construction); passing a non-None new_cover raises.

    Construction for eta.dim >= 1:
        1. Take le[i] = lifted_eta(eta)[i], the (n-1)-preimage satisfying
           δ(le[i]) = eta|_{cover[i]}. Since eta.section ⊆ cover[i], this is
           just δ(le[i]) = eta.
        2. Extend le[i] via extension_by_softness_of_cech_cochains to X.
        3. Take the coboundary of the extended (n-1)-cochain. Since δ² = 0,
           the result is automatically a cocycle on X.

    Preconditions (raises ValueError if violated):
      dim == 0 branch (inherited from the 0-cocycle specialization):
          - eta.dim == 0, eta is a cocycle
          - eta.grp is EG-valued
          - g ≡ 1 on V, g ≡ 0 on X ∖ U  (where U is eta's effective domain)
      dim >= 1 branch:
          - i is not None
          - new_cover is None (the cover cannot be changed in this branch)
          - eta.section is not None
          - eta.section ⊆ eta.cover[i]
          - eta is a Čech cocycle
          - δ(le[i]) == eta|_{cover[i]}   (validate_lifted_eta(eta, i))
          - eta.grp is EG-valued; g ≡ 1 on V, g ≡ 0 on X ∖ le[i].section
            (enforced inside extension_by_softness_of_cech_cochains)

    Parameters
    ----------
    eta : CechCochain
        An EG-valued n-cocycle.
    V : array-like of ints
        "Core" index set on which g ≡ 1.
    g : callable(idx) -> float in [0,1]
        The softening function.
    X : array-like of ints
        Target section.
    i : int or None, optional (default None)
        The cover index such that eta.section ⊆ cover[i]. Required for
        eta.dim >= 1; ignored for eta.dim == 0.
    tol : float, default 1e-9
        Tolerance for g-value preconditions.
    new_cover : numpy array of dtype=object, optional
        Only meaningful when eta.dim == 0 (passed through to
        extension_by_softness_of_zero_cocycles). For eta.dim >= 1, must be
        None — otherwise raises, since the output cover is determined by
        the coboundary construction.

    Returns
    -------
    CechCochain
        A new n-cocycle with section = X.
    """
    # ---- Dispatch on dim ----
    if eta.dim == 0:
        return extension_by_softness_of_zero_cocycles(
            eta, V, g, X, tol=tol, new_cover=new_cover,
        )

    # ---- From here on, eta.dim >= 1 ----

    # i is required
    if i is None:
        raise ValueError(
            f"extension_by_softness_of_cech_cocycles: for eta.dim = {eta.dim} "
            f">= 1, the index i is required (cannot be None). i must satisfy "
            f"eta.section ⊆ eta.cover[i]."
        )

    # new_cover has no effect here — reject it to catch user mistakes
    if new_cover is not None:
        raise ValueError(
            f"extension_by_softness_of_cech_cocycles: new_cover has no effect "
            f"for eta.dim = {eta.dim} >= 1 (the output cover is determined by "
            f"the coboundary construction and equals eta.cover). Drop the "
            f"new_cover argument or use eta.dim == 0 where it applies."
        )

    # 0. eta.section must be explicit
    if eta.section is None:
        raise ValueError(
            "extension_by_softness_of_cech_cocycles (eta.dim >= 1) requires "
            "eta.section to be an explicit index set (not None). The special "
            "case assumption eta.section ⊆ cover[i] does not make sense with "
            "an implicit 'full ambient' section."
        )

    # 1. eta.section ⊆ cover[i]
    cover_i_set = set(int(k) for k in eta.cover[i])
    section_set = set(int(k) for k in eta.section)
    extra = section_set - cover_i_set
    if extra:
        preview = sorted(extra)[:5]
        more = "" if len(extra) <= 5 else f" (+{len(extra) - 5} more)"
        raise ValueError(
            f"extension_by_softness_of_cech_cocycles: eta.section is not a "
            f"subset of eta.cover[{i}]. {len(extra)} offending index/indices; "
            f"first few: {preview}{more}."
        )

    # 2. eta must be a cocycle
    if not validate_cech_cocycle(eta):
        raise ValueError(
            "extension_by_softness_of_cech_cocycles: eta is not a cocycle "
            "(validate_cech_cocycle returned False)."
        )

    # 3. δ(le[i]) == eta|_{cover[i]} (= eta here, since section ⊆ cover[i])
    validate_lifted_eta(eta, i)

    # 4. Take le[i] as the (n-1)-preimage
    le_i = lifted_eta(eta)[i]

    # 5. Extend le[i] by softness, then take coboundary
    extended_le_i = extension_by_softness_of_cech_cochains(le_i, V, g, X, tol=tol)
    new_eta       = compute_cech_coboundary(extended_le_i)

    return new_eta


# =============================================================================
# 10. Materialization
# =============================================================================

def _materialize(eta):
    """
    Precompute eta on every standard (eta.dim+1)-simplex once and return an
    equivalent CechCochain whose func is a dict lookup. This "flattens" a
    deeply nested closure chain into O(1)-per-query, at the cost of an
    upfront pass over all standard simplices.

    Equivalent to eta as a cochain (same values everywhere), just faster
    when eta itself is layered on top of many other cochains.

    Parameters
    ----------
    eta : CechCochain

    Returns
    -------
    CechCochain
        Same dim, cover, grp, section as eta; func replaced by a cache lookup.
    """
    N = len(eta.cover)
    cache = {}
    for sigma in itertools.combinations(range(N), eta.dim + 1):
        cache[sigma] = dict(eta(sigma))   # force evaluation; copy is paranoia

    inv_op = eta.grp['inv']

    def materialized_func(simplex):
        sorted_simplex = tuple(sorted(simplex))
        base = cache[sorted_simplex]
        if simplex == sorted_simplex:
            return dict(base)
        # Non-standard: apply inv if odd permutation
        is_odd = _permutation_sign(simplex)
        if is_odd:
            return {idx: inv_op(v) for idx, v in base.items()}
        return dict(base)

    return CechCochain(
        dim     = eta.dim,
        cover   = eta.cover,
        grp     = eta.grp,
        func    = materialized_func,
        section = eta.section,
    )


# =============================================================================
# 11. Dimension lowering
# =============================================================================

def Q_lower_cech_dim_on_F_EG(eta, cover_filtration, g):
    """
    Lower the Čech dimension of an EG-valued cocycle by one, using a
    filtration of open covers and a sequence of softening functions.

    See body for the full construction. Performance: at the end of each
    iteration we materialize t_i so the next iteration's t_prev is a
    flat dict-lookup closure rather than a deeply nested chain. Without
    this, querying t_prev would recursively unfold all earlier iterations
    and give roughly O(2^i) blowup.

    Preconditions (raises ValueError if violated):
      - eta.grp is EG-valued (eta.grp['source'] is a group dict).
      - eta is a Čech cocycle.
      - eta.dim >= 1.
      - len(cover_filtration) == len(eta.cover).
      - cover_filtration[0] equals eta.cover (element-wise).

    Soft precondition (warns but does not raise):
      - eta.section is None.

    Returns a cochain (not necessarily a cocycle) of dimension eta.dim - 1,
    with the same cover and grp as eta, and section normalized to None
    if it accumulates to the full ambient domain.
    """
    # ---- Input checks ----
    source = eta.grp.get('source')
    if not isinstance(source, dict):
        raise ValueError(
            f"Q_lower_cech_dim_on_F_EG requires eta to be EG-valued "
            f"(eta.grp must be constructed via eg_tools.make_EG). "
            f"Got eta.grp with name '{eta.grp.get('name', '?')}' and source "
            f"{source!r}."
        )

    if not validate_cech_cocycle(eta):
        raise ValueError(
            "Q_lower_cech_dim_on_F_EG requires eta to be a Čech cocycle "
            "(validate_cech_cocycle returned False)."
        )

    if eta.dim < 1:
        raise ValueError(
            f"Q_lower_cech_dim_on_F_EG requires eta.dim >= 1, got eta.dim = "
            f"{eta.dim}. (Lowering dimension from 0 is not defined here.)"
        )

    if eta.section is not None:
        warnings.warn(
            "Q_lower_cech_dim_on_F_EG: eta.section is not None. The algorithm "
            "is designed for cocycles on the global section "
            "(eta.section = None); the result may not behave as expected."
        )

    if len(cover_filtration) != len(eta.cover):
        raise ValueError(
            f"Q_lower_cech_dim_on_F_EG: len(cover_filtration) = "
            f"{len(cover_filtration)} must equal len(eta.cover) = "
            f"{len(eta.cover)}."
        )

    if len(cover_filtration[0]) != len(eta.cover):
        raise ValueError(
            f"Q_lower_cech_dim_on_F_EG: cover_filtration[0] has "
            f"{len(cover_filtration[0])} open sets, but eta.cover has "
            f"{len(eta.cover)}."
        )
    for j in range(len(eta.cover)):
        if not np.array_equal(cover_filtration[0][j], eta.cover[j]):
            raise ValueError(
                f"Q_lower_cech_dim_on_F_EG: cover_filtration[0][{j}] does "
                f"not equal eta.cover[{j}] (element-wise comparison)."
            )

    # ---- Helper index-set functions ----
    cover = cover_filtration
    N = len(cover[0])

    def U_tilde(i):
        u = set()
        for j in range(i + 1):
            u.update(int(k) for k in cover[i][j])
        u_arr = np.array(sorted(u))
        return np.intersect1d(u_arr, cover[0][i + 1])

    def W_tilde(i):
        if i < 1:
            raise ValueError(f"W_tilde requires i >= 1, got {i}.")
        u = set()
        for j in range(i):
            u.update(int(k) for k in cover[i][j])
        return np.array(sorted(u))

    def V_tilde(i):
        return np.intersect1d(W_tilde(i), cover[i][i])

    # ---- Main loop ----
    le = lifted_eta(eta)
    t_prev = le[0]

    for i in range(1, N):
        t0 = time.time()
        print(f"· iter i={i} starting...")
        
        u_prev = U_tilde(i - 1)

        ti_tilde = add_cech_cochains(
            restrict_eta_to(t_prev, u_prev),
            inv_cech_cochains(restrict_eta_to(le[i], u_prev)),
        )

        ti_bar = extension_by_softness_of_cech_cocycles(
            ti_tilde, V_tilde(i), g[i], cover[0][i], i=i, tol=1e-9,
        )

        t_i = gluing_cech_cochains(
            restrict_eta_to(t_prev, W_tilde(i)),
            restrict_eta_to(
                add_cech_cochains(le[i], ti_bar),
                cover[i][i],
            ),
        )

        # Materialize t_i to flatten closure depth before the next iteration.
        print(f"· iter i={i} done in {time.time() - t0:.1f}s")
        
        t_prev = _materialize(t_i)

    # ---- Cleanup: normalize section to None if it equals the full cover ----
    full = set()
    for arr in t_prev.cover:
        full.update(int(k) for k in arr)
    if t_prev.section is not None and set(int(k) for k in t_prev.section) == full:
        t_prev.section = None
    
    print(f"· Q^{eta.dim}(-) is Done")
    return t_prev


# =============================================================================
# 12. Additional validators
# =============================================================================

def validate_lifted_eta_2(eta):
    """
    For each standard 1-simplex (i, j) with i < j, verify that

        (lifted_eta[i] | U_i ∩ U_j) - (lifted_eta[j] | U_i ∩ U_j)

    is a cocycle on U_i ∩ U_j.

    Mathematical background:
        δ(lifted_eta[i]) = eta|_{U_i}
        δ(lifted_eta[j]) = eta|_{U_j}
    Restricted to U_i ∩ U_j, both coboundaries equal eta|_{U_i ∩ U_j}, so
    the difference δ(lifted_eta[i] - lifted_eta[j])|_{U_i ∩ U_j} = 0, i.e.
    the restricted difference is a cocycle.

    Only standard 1-simplices (i < j) are checked: the (j, i) case gives
    the negative cochain, which is a cocycle iff the (i, j) case is.

    Parameters
    ----------
    eta : CechCochain

    Returns
    -------
    True if the restricted difference is a cocycle for every standard 1-simplex.

    Raises
    ------
    ValueError
        If validation fails for some (i, j), with a message identifying
        which pair.
    """
    if not VALIDATE:
        return True

    le = lifted_eta(eta)
    N  = len(eta.cover)

    for i, j in itertools.combinations(range(N), 2):
        U_ij = compute_intersection((i, j), eta.cover)

        le_i_on_U_ij = restrict_eta_to(le[i], U_ij)
        le_j_on_U_ij = restrict_eta_to(le[j], U_ij)

        diff = add_cech_cochains(
            le_i_on_U_ij,
            inv_cech_cochains(le_j_on_U_ij),
        )

        if not validate_cech_cocycle(diff):
            raise ValueError(
                f"validate_lifted_eta_2 failed for (i, j) = ({i}, {j}): "
                f"(lifted_eta[{i}] - lifted_eta[{j}]) | U_{i} ∩ U_{j} "
                f"is not a cocycle."
            )

    return True


def validate_Q_lower_cech_on_F_EG(i_sharp_eta, cover_filtration, g):
    """
    Test whether Q_lower_cech_dim_on_F_EG inverts the Čech coboundary on
    i_sharp(eta), i.e. whether

        δ(  Q( i_sharp_eta )  )   ==   i_sharp_eta

    as Čech cocycles over F_EG. If this equality holds, Q is genuinely a
    section of δ at i_sharp_eta.

    NOTE: failure does not necessarily mean Q is wrong — the two cocycles
    might only be homotopic rather than equal. Detecting that case would
    require comparing cohomology classes, which is not implemented here.
    This function only checks pointwise equality.

    Parameters
    ----------
    i_sharp_eta : CechCochain
        A cocycle in group EG.
    cover_filtration : sequence of covers
        Passed to Q_lower_cech_dim_on_F_EG.
    g : list / sequence
        Passed to Q_lower_cech_dim_on_F_EG.

    Returns
    -------
    bool
        True if δ(Q(i_sharp(eta))) and i_sharp(eta) are equal everywhere;
        False otherwise (with a diagnostic message printed).
    """
    Q_eta = Q_lower_cech_dim_on_F_EG(i_sharp_eta, cover_filtration, g)

    if not VALIDATE:
        return Q_eta

    delta_Q_eta = compute_cech_coboundary(Q_eta)

    try:
        cech_cochains_equal(delta_Q_eta, i_sharp_eta, func=True, section=True)
    except ValueError as e:
        print("✗ δ(Q(i_sharp(eta)))  ≠  i_sharp(eta) :")
        print(str(e))
        print()
        print("(This may still mean the two are homotopic / represent the "
              "same cohomology class, which is not checked here.)")
        raise ValueError(
                "validate_Q_lower_cech_on_F_EG fails: δ(Q(i_sharp(eta)))  ≠  i_sharp(eta)"
            )

    print("✓ δ(Q(i_sharp(eta)))  ==  i_sharp(eta)   as Čech cocycles over F_EG.")
    return Q_eta
