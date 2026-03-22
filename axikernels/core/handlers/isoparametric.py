"""
Reference-space isoparametric interpolation for AxiSEM3D element output.

AxiSEM3D stores element output at 9 physical (s, z) points arranged as a
3×3 tensor-product grid in reference coordinates (ξ, η) ∈ [-1, 1]².

Storage order: ipnt = ipol × 3 + jpol  (ipol → ξ outer, jpol → η inner)

Point layout::

    ipnt:  0  1  2    ← ipol=0, jpol=0,1,2  (ξ = xi_nodes[0])
           3  4  5    ← ipol=1, jpol=0,1,2  (ξ = xi_nodes[1])
           6  7  8    ← ipol=2, jpol=0,1,2  (ξ = xi_nodes[2])

Corner indices: [0, 2, 6, 8].  LEFT edge (ξ=-1): indices [0, 1, 2].

Reference abscissae depend on element type:
  - Non-axial (GLL × GLL):  ξ_nodes = [-1, 0, 1],          η_nodes = [-1, 0, 1]
  - Axial     (GLJ × GLL):  ξ_nodes = [-1, 0.1323..., 1],  η_nodes = [-1, 0, 1]

The full 5-point nPol=4 sets (stored subset uses indices [0, 2, 4]):
  GLL: [-1, -0.6547, 0, +0.6547, +1]
  GLJ: [-1, -0.5078, +0.1323, +0.7088, +1]

References
----------
- AxiSEM3D source: src/core/element/mapping/Mapping.hpp
- Komatitsch & Tromp (2002): spectral-element method for the Earth
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Constants: stored 3-point subsets of the 5-point GLL/GLJ sets
# (indices [0, 2, 4] of the full nPol=4 set)
# ---------------------------------------------------------------------------

#: Gauss-Lobatto-Legendre 3-point subset: [-1, 0, 1]
GLL_SUBSET: np.ndarray = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

#: Gauss-Lobatto-Jacobi 3-point subset (non-axial partner nodes for axial elements)
GLJ_SUBSET: np.ndarray = np.array([-1.0, 0.132300820777, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. One-dimensional Lagrange basis
# ---------------------------------------------------------------------------

def lagrange_1d(t: float, nodes: np.ndarray, i: int) -> float:
    r"""Evaluate the i-th Lagrange basis polynomial at point *t*.

    Given *n* distinct nodes :math:`x_0, \ldots, x_{n-1}`, the i-th basis
    polynomial is:

    .. math::

        L_i(t) = \prod_{j \ne i} \frac{t - x_j}{x_i - x_j}

    Parameters
    ----------
    t : float
        Evaluation point.
    nodes : np.ndarray, shape (n,)
        The *n* distinct interpolation nodes.
    i : int
        Index of the basis polynomial to evaluate (0 ≤ i < n).

    Returns
    -------
    float
        Value of :math:`L_i(t)`.
    """
    val = 1.0
    xi = nodes[i]
    for j, xj in enumerate(nodes):
        if j != i:
            val *= (t - xj) / (xi - xj)
    return float(val)


def lagrange_weights_1d(t: float, nodes: np.ndarray) -> np.ndarray:
    r"""Evaluate all n Lagrange basis weights at point *t*.

    Returns the vector :math:`[L_0(t), L_1(t), \ldots, L_{n-1}(t)]`.

    The weights form a partition of unity: :math:`\sum_i L_i(t) = 1`.

    Parameters
    ----------
    t : float
        Evaluation point.
    nodes : np.ndarray, shape (n,)
        The *n* distinct interpolation nodes.

    Returns
    -------
    np.ndarray, shape (n,)
        Lagrange basis weights at *t*.
    """
    n = len(nodes)
    return np.array([lagrange_1d(t, nodes, i) for i in range(n)], dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. Axial detection and reference abscissae
# ---------------------------------------------------------------------------

def compute_min_edge_length(element_coords_9: np.ndarray) -> float:
    r"""Compute the minimum edge length from the 4 corner nodes.

    Corner layout (ipnt = ipol×3 + jpol):
      - Index 0: corner (ξ=-1, η=-1)
      - Index 2: corner (ξ=-1, η=+1)
      - Index 6: corner (ξ=+1, η=-1)
      - Index 8: corner (ξ=+1, η=+1)

    The 4 edges are: 0-2 (left), 6-8 (right), 0-6 (bottom), 2-8 (top).

    Parameters
    ----------
    element_coords_9 : np.ndarray, shape (9, 2)
        The 9 stored (s, z) coordinates in ipnt order.

    Returns
    -------
    float
        Minimum of the 4 corner-edge lengths.
    """
    corners = element_coords_9[[0, 2, 6, 8]]  # shape (4, 2)
    c0, c2, c6, c8 = corners[0], corners[1], corners[2], corners[3]
    edges = [
        np.linalg.norm(c2 - c0),   # left   edge  (ξ=-1): 0→2
        np.linalg.norm(c8 - c6),   # right  edge  (ξ=+1): 6→8
        np.linalg.norm(c6 - c0),   # bottom edge  (η=-1): 0→6
        np.linalg.norm(c8 - c2),   # top    edge  (η=+1): 2→8
    ]
    return float(np.min(edges))


def detect_axial(element_coords_9: np.ndarray, tol: float = 1e-3) -> bool:
    r"""Detect whether an element is axial (one edge on the symmetry axis s=0).

    An element is axial if the three nodes on its LEFT edge (ipol=0,
    jpol=0,1,2; indices 0,1,2 in ipnt order) all satisfy:

    .. math::

        |s_k| < \text{tol} \times l_{\min}

    where :math:`l_{\min}` is the minimum corner-edge length computed by
    :func:`compute_min_edge_length`.

    Parameters
    ----------
    element_coords_9 : np.ndarray, shape (9, 2)
        The 9 stored (s, z) coordinates in ipnt order.
    tol : float, optional
        Relative tolerance factor.  Default: 1e-3.

    Returns
    -------
    bool
        ``True`` if the element is axial; ``False`` otherwise.
    """
    min_edge = compute_min_edge_length(element_coords_9)
    threshold = tol * min_edge
    left_edge_s = element_coords_9[[0, 1, 2], 0]  # s-values on LEFT edge
    return bool(np.all(np.abs(left_edge_s) < threshold))


def reference_abscissae(axial: bool) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the stored 3-point reference abscissae for an element.

    Parameters
    ----------
    axial : bool
        ``True`` for an axial element (GLJ × GLL);
        ``False`` for a non-axial element (GLL × GLL).

    Returns
    -------
    xi_nodes : np.ndarray, shape (3,)
        Reference abscissae in the ξ-direction.
    eta_nodes : np.ndarray, shape (3,)
        Reference abscissae in the η-direction (always GLL).
    """
    if axial:
        return GLJ_SUBSET.copy(), GLL_SUBSET.copy()
    return GLL_SUBSET.copy(), GLL_SUBSET.copy()


# ---------------------------------------------------------------------------
# 3. Isoparametric forward map and Jacobian
# ---------------------------------------------------------------------------

def forward_map_9node(
    xi: float,
    eta: float,
    element_coords_9: np.ndarray,
    xi_nodes: np.ndarray,
    eta_nodes: np.ndarray,
) -> tuple[float, float]:
    r"""Evaluate the 9-node isoparametric forward map :math:`F(\xi, \eta)`.

    .. math::

        F(\xi, \eta)
            = \sum_{i=0}^{2} \sum_{j=0}^{2}
              L_i(\xi)\, L_j(\eta)\, \mathbf{X}_{ij}

    where :math:`\mathbf{X}_{ij} = (s_{ij}, z_{ij})` are the stored
    physical coordinates at tensor-product node (i, j), stored at index
    ``ipnt = i*3 + j`` in *element_coords_9*.

    Parameters
    ----------
    xi : float
        Reference ξ coordinate.
    eta : float
        Reference η coordinate.
    element_coords_9 : np.ndarray, shape (9, 2)
        Stored (s, z) coordinates in ipnt = ipol*3+jpol order.
    xi_nodes : np.ndarray, shape (3,)
        Reference abscissae in ξ (GLL or GLJ).
    eta_nodes : np.ndarray, shape (3,)
        Reference abscissae in η (always GLL).

    Returns
    -------
    s : float
    z : float
    """
    wx = lagrange_weights_1d(xi, xi_nodes)   # shape (3,)
    wy = lagrange_weights_1d(eta, eta_nodes)  # shape (3,)

    # element_coords_9 reshaped to (ipol, jpol, 2)
    coords_2d = element_coords_9.reshape(3, 3, 2)  # [ipol, jpol, (s,z)]

    # F = sum_i sum_j wx[i] * wy[j] * X[i,j]
    # = (wx @ coords_2d @ wy) using matrix form
    # coords_2d[:, :, c] is the (3,3) matrix of the c-th coordinate
    s = float(wx @ coords_2d[:, :, 0] @ wy)
    z = float(wx @ coords_2d[:, :, 1] @ wy)
    return s, z


def jacobian_9node(
    xi: float,
    eta: float,
    element_coords_9: np.ndarray,
    xi_nodes: np.ndarray,
    eta_nodes: np.ndarray,
) -> np.ndarray:
    r"""Compute the 2×2 Jacobian :math:`\partial(s,z)/\partial(\xi,\eta)`.

    .. math::

        J = \begin{pmatrix}
              \partial s / \partial \xi  & \partial s / \partial \eta \\
              \partial z / \partial \xi  & \partial z / \partial \eta
            \end{pmatrix}

    Computed analytically from derivatives of the Lagrange basis:

    .. math::

        \frac{\partial F}{\partial \xi}
            = \sum_i \sum_j L_i'(\xi)\, L_j(\eta)\, \mathbf{X}_{ij}

        \frac{\partial F}{\partial \eta}
            = \sum_i \sum_j L_i(\xi)\, L_j'(\eta)\, \mathbf{X}_{ij}

    The Lagrange derivative :math:`L_i'(t)` is computed analytically via
    :func:`_lagrange_deriv_1d` using the barycentric formula:

    .. math::

        L_i'(t) = L_i(t) \sum_{j \ne i} \frac{1}{t - x_j}

    A finite-difference fallback is used only when *t* coincides with a
    node (relative distance < 1e-14) to avoid division by zero at that
    degenerate point.

    Parameters
    ----------
    xi, eta : float
        Reference coordinates at which to evaluate the Jacobian.
    element_coords_9 : np.ndarray, shape (9, 2)
        Stored (s, z) coordinates.
    xi_nodes, eta_nodes : np.ndarray, shape (3,)
        Reference abscissae.

    Returns
    -------
    J : np.ndarray, shape (2, 2)
        Jacobian matrix.
    """
    wx = lagrange_weights_1d(xi, xi_nodes)
    wy = lagrange_weights_1d(eta, eta_nodes)
    dwx = _lagrange_deriv_1d(xi, xi_nodes)
    dwy = _lagrange_deriv_1d(eta, eta_nodes)

    coords_2d = element_coords_9.reshape(3, 3, 2)  # [ipol, jpol, (s,z)]

    J = np.zeros((2, 2), dtype=np.float64)
    for c in range(2):
        X = coords_2d[:, :, c]  # (3, 3)
        J[c, 0] = float(dwx @ X @ wy)   # ∂(s or z)/∂ξ
        J[c, 1] = float(wx @ X @ dwy)   # ∂(s or z)/∂η
    return J


def _lagrange_deriv_1d(t: float, nodes: np.ndarray) -> np.ndarray:
    r"""Evaluate derivatives :math:`L_i'(t)` for all i at point *t*.

    Uses a numerically stable formulation:

    .. math::

        L_i'(t) = L_i(t) \sum_{j \ne i} \frac{1}{t - x_j}

    When *t* coincides with node *k* this formula degenerates; in that
    case we use the exact expression:

    .. math::

        L_i'(x_k)
            = \begin{cases}
                \displaystyle\prod_{j \ne k} \frac{x_k - x_j}{\,?} & \text{handled below}
              \end{cases}

    evaluated via a small finite-difference ``h = 1e-7`` perturbation
    only at exact node hits (relative distance < 1e-14).

    Parameters
    ----------
    t : float
        Evaluation point.
    nodes : np.ndarray, shape (n,)
        Interpolation nodes.

    Returns
    -------
    dw : np.ndarray, shape (n,)
        Derivative weights.
    """
    n = len(nodes)
    dists = np.abs(t - nodes)
    # Check if t is exactly (or nearly) one of the nodes
    hit_idx = np.where(dists < 1e-14 * max(1.0, abs(t)))[0]

    if len(hit_idx) > 0:
        # Finite-difference fallback at node hit to avoid division by zero
        h = 1e-7
        w_p = lagrange_weights_1d(t + h, nodes)
        w_m = lagrange_weights_1d(t - h, nodes)
        return (w_p - w_m) / (2.0 * h)

    # General case: L_i'(t) = L_i(t) * sum_{j≠i} 1/(t - x_j)
    w = lagrange_weights_1d(t, nodes)
    dw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j != i:
                s += 1.0 / (t - nodes[j])
        dw[i] = w[i] * s
    return dw


# ---------------------------------------------------------------------------
# 4. Newton inverse mapping
# ---------------------------------------------------------------------------

def newton_inverse(
    s_target: float,
    z_target: float,
    element_coords_9: np.ndarray,
    xi_nodes: np.ndarray,
    eta_nodes: np.ndarray,
    max_iter: int = 10,
    tolerance: float = 1e-9,
) -> tuple[float, float, bool, bool]:
    r"""Newton inverse mapping: physical (s, z) → reference (ξ, η).

    Solves :math:`F(\xi, \eta) = (s_t, z_t)` by damped Newton iteration
    following AxiSEM3D's ``Mapping.hpp``:

    1. Initialise :math:`(\xi, \eta) = (0, 0)`.
    2. Compute :math:`\Delta \mathbf{x} = (s_t, z_t) - F(\xi, \eta)`.
    3. If :math:`\|\Delta \mathbf{x}\| < \varepsilon_{sz}`, break.
    4. Update :math:`(\xi, \eta) \mathrel{+}= J^{-1} \Delta \mathbf{x}`.

    Convergence threshold:
    :math:`\varepsilon_{sz} = \text{tolerance} \times l_{\min}`.

    The ``inside`` flag is set to ``True`` when:
    :math:`-\text{bound} < \xi, \eta < \text{bound}`, where
    :math:`\text{bound} = 1 + 20 \times \text{tolerance}`.

    Parameters
    ----------
    s_target, z_target : float
        Target physical coordinates.
    element_coords_9 : np.ndarray, shape (9, 2)
        Stored (s, z) node coordinates.
    xi_nodes, eta_nodes : np.ndarray, shape (3,)
        Reference abscissae for this element.
    max_iter : int, optional
        Maximum Newton iterations.  Default: 10.
    tolerance : float, optional
        Relative convergence tolerance.  Default: 1e-9.

    Returns
    -------
    xi : float
        Converged reference ξ coordinate.
    eta : float
        Converged reference η coordinate.
    converged : bool
        ``True`` if the residual fell below the threshold within *max_iter*.
    inside : bool
        ``True`` if *converged* is ``True`` and :math:`(\xi, \eta)` lies
        within the element bounds, i.e.
        :math:`-\text{bound} \leq \xi, \eta \leq \text{bound}` where
        :math:`\text{bound} = 1 + 20 \times \text{tolerance}`.
        Always ``False`` when *converged* is ``False``.
    """
    min_edge = compute_min_edge_length(element_coords_9)
    eps_sz = tolerance * min_edge

    xi = 0.0
    eta = 0.0
    converged = False

    for _ in range(max_iter):
        s_calc, z_calc = forward_map_9node(xi, eta, element_coords_9, xi_nodes, eta_nodes)
        ds = s_target - s_calc
        dz = z_target - z_calc
        residual = np.sqrt(ds * ds + dz * dz)
        if residual < eps_sz:
            converged = True
            break
        J = jacobian_9node(xi, eta, element_coords_9, xi_nodes, eta_nodes)
        try:
            delta = np.linalg.solve(J, np.array([ds, dz], dtype=np.float64))
        except np.linalg.LinAlgError:
            # Singular Jacobian — cannot proceed
            break
        xi += delta[0]
        eta += delta[1]

    bound = 1.0 + 20.0 * tolerance
    if not converged:
        inside = False
    else:
        inside = bool((-bound <= xi <= bound) and (-bound <= eta <= bound))
    return float(xi), float(eta), converged, inside


# ---------------------------------------------------------------------------
# 5. Interpolation weight vector (9-node)
# ---------------------------------------------------------------------------

def interpolation_weights_9node(
    xi: float,
    eta: float,
    xi_nodes: np.ndarray,
    eta_nodes: np.ndarray,
) -> np.ndarray:
    r"""Compute the 9-element interpolation weight vector.

    The weight for physical node at ``ipnt = ipol*3 + jpol`` is:

    .. math::

        w_{\text{ipnt}} = L_{\text{ipol}}(\xi)\, L_{\text{jpol}}(\eta)

    The result matches the storage order used elsewhere in axikernels
    (``ipol`` → ξ outer loop, ``jpol`` → η inner loop).

    Satisfies partition of unity: :math:`\sum_k w_k = 1`.

    Parameters
    ----------
    xi : float
        Reference ξ coordinate.
    eta : float
        Reference η coordinate.
    xi_nodes : np.ndarray, shape (3,)
        Reference abscissae in ξ.
    eta_nodes : np.ndarray, shape (3,)
        Reference abscissae in η.

    Returns
    -------
    weights : np.ndarray, shape (9,)
        Flattened outer-product weight vector in ipnt = ipol*3+jpol order.
    """
    wx = lagrange_weights_1d(xi, xi_nodes)    # shape (3,): L_ipol(ξ)
    wy = lagrange_weights_1d(eta, eta_nodes)  # shape (3,): L_jpol(η)
    # Outer product: w[ipol, jpol] = wx[ipol] * wy[jpol]
    # Flatten in C order → index ipol*3 + jpol
    return np.outer(wx, wy).ravel().astype(np.float64)


# ---------------------------------------------------------------------------
# 6. KDTree-based element search
# ---------------------------------------------------------------------------

def build_element_kdtree(
    all_element_coords: np.ndarray,
) -> tuple[KDTree, np.ndarray]:
    r"""Build a :class:`scipy.spatial.KDTree` on element centres.

    The centre of each element is defined as the mean of its 9 stored
    physical coordinates.

    Parameters
    ----------
    all_element_coords : array-like, shape (n_elements, 9, 2)
        Physical (s, z) coordinates of all elements, each with 9 nodes.

    Returns
    -------
    kdtree : scipy.spatial.KDTree
        KDTree built on element centres, shape (n_elements, 2).
    centers : np.ndarray, shape (n_elements, 2)
        The element centre coordinates used to build the tree.
    """
    all_element_coords = np.asarray(all_element_coords, dtype=np.float64)
    centers = np.mean(all_element_coords, axis=1)  # shape (n_elements, 2)
    kdtree = KDTree(centers)
    return kdtree, centers


def find_containing_element(
    s: float,
    z: float,
    all_element_coords: np.ndarray,
    kdtree: KDTree,
    k: int = 20,
) -> tuple[int, float, float]:
    r"""Find the element that contains physical point (s, z).

    Queries the KDTree for the *k* nearest element centres, then applies
    Newton inverse mapping (:func:`newton_inverse`) to each candidate in
    order of distance until one reports ``converged and inside``.

    Parameters
    ----------
    s, z : float
        Target physical coordinates.
    all_element_coords : np.ndarray, shape (n_elements, 9, 2)
        Physical (s, z) coordinates of all elements.
    kdtree : scipy.spatial.KDTree
        Pre-built KDTree from :func:`build_element_kdtree`.
    k : int, optional
        Number of nearest neighbours to check.  Clamped to
        ``len(all_element_coords)``.  Default: 20.

    Returns
    -------
    element_index : int
        Index of the containing element, or ``-1`` if none found.
    xi : float
        Reference ξ coordinate inside the element, or ``nan``.
    eta : float
        Reference η coordinate inside the element, or ``nan``.
    """
    n_elem = len(all_element_coords)
    k = min(k, n_elem)

    _, candidate_indices = kdtree.query([s, z], k=k)
    # kdtree.query returns a scalar when k=1; normalise to 1-D array
    candidate_indices = np.atleast_1d(candidate_indices)

    for elem_idx in candidate_indices:
        coords_9 = all_element_coords[elem_idx]
        axial = detect_axial(coords_9)
        xi_nodes, eta_nodes = reference_abscissae(axial)
        xi, eta, converged, inside = newton_inverse(s, z, coords_9, xi_nodes, eta_nodes)
        if converged and inside:
            return int(elem_idx), xi, eta

    return -1, np.nan, np.nan


def find_containing_elements_batch(
    points_sz: np.ndarray,
    all_element_coords: np.ndarray,
    kdtree: KDTree,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Find containing elements for a batch of physical points.

    Vectorised wrapper around :func:`find_containing_element` that
    processes each row of *points_sz* independently.

    Parameters
    ----------
    points_sz : np.ndarray, shape (n_points, 2)
        Array of (s, z) query coordinates.
    all_element_coords : np.ndarray, shape (n_elements, 9, 2)
        Physical (s, z) coordinates of all elements.
    kdtree : scipy.spatial.KDTree
        Pre-built KDTree from :func:`build_element_kdtree`.
    k : int, optional
        Number of nearest neighbours passed to each single-point search.
        Default: 20.

    Returns
    -------
    element_indices : np.ndarray, shape (n_points,), dtype int
        Index of the containing element per point, or ``-1``.
    xi_arr : np.ndarray, shape (n_points,)
        Reference ξ per point, or ``nan`` for unmatched points.
    eta_arr : np.ndarray, shape (n_points,)
        Reference η per point, or ``nan`` for unmatched points.
    """
    points_sz = np.asarray(points_sz, dtype=np.float64)
    n_points = len(points_sz)
    element_indices = np.empty(n_points, dtype=np.intp)
    xi_arr = np.empty(n_points, dtype=np.float64)
    eta_arr = np.empty(n_points, dtype=np.float64)

    for i in range(n_points):
        idx, xi, eta = find_containing_element(
            points_sz[i, 0], points_sz[i, 1], all_element_coords, kdtree, k=k
        )
        element_indices[i] = idx
        xi_arr[i] = xi
        eta_arr[i] = eta

    return element_indices, xi_arr, eta_arr
