from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET
from .schemas import NEUMANN

# TODO: begin implementing neohook material
# TODO: compute element residual vector
# TODO: implement newton-raphson solver

def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Map a node and local DOF index to the corresponding global DOF."""
    return node * dof_per_node + local_dof

def gauss_points_weights(n_gauss: int = 2):
    """Return Gauss points and weights for 1D integration on [-1, 1]
        Takes Inputs:
            n_gauss: number of gauss points
        Returns:
            pts: gauss points
            wts: weights at associated gauss points"""
    if n_gauss == 1:
        # 1-point quadrature
        pts = np.array([0.0])
        wts = np.array([2.0])
    elif n_gauss == 2:
        # 2-point quadrature (default)
        pts = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        wts = np.array([1.0, 1.0])
    elif n_gauss == 3:
        # 3-point quadrature (more accurate for nonlinear problems)
        pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        wts = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError(f"Unsupported number of Gauss points: {n_gauss}")
    return pts, wts

def elem_stiff(
    material: dict[str, Any],
    xe: NDArray[np.float64],
    props: dict[str, Any],
    ue: NDArray[np.float64] | None = None,
    n_gauss: int = 2,
) -> NDArray[np.float64]:
    """Compute the element stiffness matrix for a 1D bar or Euler-Bernoulli beam.

    Parameters
    ------------------
    material: dict
        Material data, must contain 'type' and 'parameters' including 'E'.
    xe: (2,1) ndarray
        Element nodal coordinates ``[[x1], [x2]]``.
    props: dict
        Element property dictionary containing ``'properties'``.  For bars this
        must include ``'area'``; for Euler-Bernoulli beams it must include
        ``'inertia'`` (and may also include ``'area'``).
    ue: (2,) or (4,) ndarray, optional
        Element nodal displacement vector.  Not used for linear materials but
        kept for signature compatibility.
    n_gauss: int, optional
        Number of Gauss integration points for bars (default 2).

    Returns
    ---------
    ke: ndarray
        Element stiffness matrix.  Shape is ``(2, 2)`` for bars and ``(4, 4)``
        for Euler-Bernoulli beams.

    Notes
    -------
    - Linear elastic bar and Euler-Bernoulli beam formulations are supported.
    - Support for Neo-Hookean materials is retained for bar elements.
    """

    mat_type = material["type"]
    params = material["parameters"]
    properties = props.get("properties", {})

    x1, x2 = xe[0, 0], xe[1, 0]
    L = x2 - x1
    if np.isclose(L, 0.0):
        raise ValueError("Zero Length Element Detected")

    # Euler-Bernoulli beam bending stiffness (2 DOF per node: w, theta).
    if "inertia" in properties:
        E = params["E"]
        I = properties["inertia"]
        factor = E * I / (L**3)
        return factor * np.array(
            [
                [12.0, 6.0 * L, -12.0, 6.0 * L],
                [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
                [-12.0, -6.0 * L, 12.0, -6.0 * L],
                [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
            ]
        )

    # Default to axial bar formulation (1 DOF per node).
    A = properties["area"]

    dN_dksi = np.array([[-0.5, 0.5]])
    pts, wts = gauss_points_weights(n_gauss)
    ke = np.zeros((2, 2))

    for ksi, w in zip(pts, wts):
        dx_dksi = np.dot(dN_dksi, xe)
        dksi_dx = 1 / dx_dksi.item()
        dN_dx = dN_dksi * dksi_dx
        B = dN_dx
        J = dx_dksi

        if mat_type == "ELASTIC":
            E = params["E"]
            ke += B.T @ B * A * E * J * w

        elif mat_type == "NEOHOOK":
            E_lin = params.get("E", params["mu"])
            ke += (B.T @ B) * A * E_lin * J * w

        else:
            raise ValueError("Unsupported material type: {mat_type}")

    return ke

def elem_int_force(props: dict[str, Any], 
                   material: dict[str, Any],
                   xe: NDArray[np.float64], 
                   ue: NDArray[np.float64]) -> float:
    """Compute the internal axial force for a 1D bar element
     Parameters
    ------------
    props: dict
        element properties dictionary containing 'properties': {'area': A}
    material: dict
        material dictionary containing 'type' and 'parameters' including 'E'
    xe: (2,1) ndarray
        element nodal coordinates [[x1], [x2]]
    ue: (2, ) ndarray
        element nodal displacement [u1, u2]
    
     Returns
    ---------
    N: float
        internal axial force in the element (positive in tension)
    
     Notes
    -------
    - Currently supports linear elastic materials, support for neohookean materials is planned"""

    mat_type = material["type"]
    E = material["parameters"].get("E", None)
    mu = material["parameters"].get("mu", None)
    lam = material["parameters"].get("lam", None)
    A = props["properties"]["area"]

    x1, x2 = xe[:, 0]
    L = x2 - x1
    strain = (ue[1] - ue[0]) / L
    F = 1.0 + strain

    if mat_type == "ELASTIC":
        stress = E * strain
        N = A * stress

    elif mat_type == "NEOHOOK":
        stress = mu * (F - 1.0/F) + lam * (np.log(F) / F)
        N = A * stress
    else:
        raise ValueError("Unsupported material type: {mat_type}")
    
    return N
 
def elem_ext_force(props: dict[str, Any], 
                   material: dict[str, Any],
                   xe: NDArray[np.float64],
                   dload: list[dict],
                   g: float = 9.81) -> NDArray[np.float64]:
    """Compute the external force for a 1D bar element
     Parameters
    ------------
    props: dict
        element properties dictionary containing 'properties': {'area': A}
    material: dict
        material dictionary containing 'parameters' including density 'rho'
    dload: dict
        distributed load description containg 'value' (q)
    xe: (2,1) ndarray
        element nodal coordinates [[x1, x2]]
    g: float, optional
        gravitational acceleration (default 9.81)
    
     Returns
    ---------
    q_ext: (2, ) ndarray
        external equivalent nodal forces from q and graviational body forces
    
     Notes
    -------
    - Assumes constant distributed load 'q'
    - body force is assumed to be gravitational"""

    A = props["properties"]["area"]
    q = float(dload["value"])
    rho = float(material.get("density", 0.0))

    if dload["type"] not in ("BX", "GRAV"):
        raise ValueError(f"Unsupported distributed load type: {dload['type']}")

    x1, x2 = xe[:,0]
    L = x2 - x1

    q_ext = q * (L / 6)

    return q_ext

def newton_solve(
        coords: NDArray[np.float64],
        blocks: list[dict],
        bcs: list[dict],
        dloads: list[dict],
        materials: dict[str, Any],
        block_elem_map: dict[int, tuple[int, int]],
        max_iters: int = 100,
        tol: float = 1e-10,
):
    """Iteratively solve for displacements using a Newton–Raphson procedure."""

    dof_per_node = 1
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node

    prescribed = {}
    for bc in bcs:
        if "node" in bc and "nodes" not in bc:
            bc["nodes"] = [bc["node"]]
    
        if "dof" in bc and "local_dof" not in bc:
            bc["local_dof"] = bc["dof"]

        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed[I] = bc["value"]
    
    prescribed_dofs = np.array(list(prescribed.keys()), dtype=int)
    prescribed_vals = np.array(list(prescribed.values()), dtype=float)

    all_dofs = np.arange(num_dof)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    u = np.zeros(num_dof, dtype=float)

    for iter in range(1, max_iters + 1):
        R = np.zeros(num_dof, dtype=float)
        Kt = np.zeros((num_dof, num_dof), dtype=float)

        for elem_id, (blk_i, loc_i) in block_elem_map.items():
            block = blocks[blk_i]
            nodes = block["connect"][loc_i]
            xe = coords[nodes]
            props = block["element"]
            material = materials[block["material"]]
            eft = np.array([n * dof_per_node for n in nodes], dtype=int)
            ue = u[eft]

            ke = elem_stiff(material, xe, props, ue)
            Kt[np.ix_(eft, eft)] += np.asarray(ke, dtype=float)

            q_int = np.asarray(ke @ ue, dtype=float).reshape(-1)
            R[eft] += q_int

        Fext = np.zeros(num_dof, dtype=float)

        for bc in bcs:
            if "node" in bc and "nodes" not in bc:
                bc["nodes"] = [bc["node"]]

            if "dof" in bc and "local_dof" not in bc:
                bc["local_dof"] = bc["dof"]

            if bc["type"] == NEUMANN:
                for n in bc["nodes"]:
                    I = global_dof(n, bc["local_dof"], dof_per_node)
                    Fext[I] += bc["value"]

        for dload in dloads:
            for eid in dload["elements"]:
                blk_idx, loc_idx = block_elem_map[eid]
                block = blocks[blk_idx]
                material = materials[block["material"]]
                props = block["element"]
                nodes = block["connect"][loc_idx]

                xe = coords[nodes]
                qext = elem_ext_force(props, material, xe, dload, g=9.81)
                eft = np.array([n * dof_per_node for n in nodes], dtype=int)
                qvec = np.array([qext, qext], dtype=float)
                Fext[eft] += qvec

        R -= Fext

        R_reduced = R[free_dofs]
        Kt_ff = Kt[np.ix_(free_dofs, free_dofs)]

        residual_norm = np.linalg.norm(R_reduced)

        if residual_norm < tol:
            return {
                "dofs": u,
                "u": u,
                "num_iter": iter,
                "converged": True,
                "stiff": Kt,
                "force": Fext,
            }

        du_f = np.linalg.solve(Kt_ff, -R_reduced)
        u[free_dofs] += du_f
        u[prescribed_dofs] = prescribed_vals

        R_post = (Kt @ u) - Fext
        if np.linalg.norm(R_post[free_dofs]) < tol:
            return {
                "dofs": u,
                "u": u,
                "num_iter": iter,
                "converged": True,
                "stiff": Kt,
                "force": Fext,
            }

    return {
        "dofs": u,
        "u": u,
        "num_iter": max_iters,
        "residual_norm": residual_norm,
        "stiff": Kt,
        "force": Fext,
        "converged": False,
    }

def first_fe_code(
    coords: NDArray[np.float64],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:

    """Wrapper for invoking the Newton solver on a 1D finite element model."""

    return newton_solve(
        coords=coords,
        blocks=blocks,
        bcs=bcs,
        dloads=dloads,
        materials=materials,
        block_elem_map=block_elem_map,
    )