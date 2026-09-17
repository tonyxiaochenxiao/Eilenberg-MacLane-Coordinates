"""
sheaves.py
==========

Utilities for sheaves of groups on index sets.

Given a group G (in the eg_tools convention) and an index set U (a subset
of the data indices), the sheaf F_G assigns to U the space of maps
    F_G(U) := Maps(U, G) = { f : U -> G }.
An element of F_G(U) is called a *section of F_G over U*.

This module provides a `SheafSection` dataclass that encapsulates such an
element: a function U -> G together with its domain U and its target group G.
"""

from dataclasses import dataclass
from typing import Any, Callable


# =============================================================================
# SheafSection
# =============================================================================

@dataclass
class SheafSection:
    """
    A section of the sheaf F_G over a domain U.

    Concretely: an element f in F_G(U) = Maps(U, G), represented by the pair
    (domain, func) along with the target group grp.

    Attributes
    ----------
    grp : dict
        The target group G (as a group dict in the eg_tools convention:
        must have keys 'name', 'grplaw', 'identity', 'inv', 'eq').
    func : callable
        A function taking a single index (int) and returning a group element.
        Its behaviour on indices outside `domain` is never invoked (calls
        with such indices are rejected by __call__ before reaching func).
    domain : array-like of ints
        The index set U on which this section is defined. Can be a list,
        tuple, numpy array, etc.

    Calling a SheafSection instance on an index verifies that the index
    is in `domain` (raising ValueError otherwise) and then delegates to
    `func`.
    """
    grp    : dict
    func   : Callable
    domain : Any

    def __post_init__(self):
        # Cache domain as a set of Python ints for fast, reliable membership
        # checks. Keeping the original `domain` unchanged preserves whatever
        # type the user passed in (e.g. a numpy array).
        self._domain_set = set(int(i) for i in self.domain)

    def __call__(self, idx):
        if int(idx) not in self._domain_set:
            raise ValueError(
                f"Index {idx} is not in this SheafSection's domain "
                f"(domain has {len(self._domain_set)} indices)."
            )
        return self.func(idx)
