from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET
from .schemas import NEUMANN

# Gauss Quadrature Helper
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

# Element Level Modularization
def elem_stiff(A: float, materials: dict, xe: NDArray[float], n_gauss: int = 2) -> NDArray[float]: # type: ignore

    mat_type = materials['type']
    params = materials["parameters"]

    x1, x2 = xe[0, 0], xe[1, 0]
    L = x2 - x1
    if np.isclose(L, 0.0):
        raise ValueError("Zero-length element detected.")

    dN_dxi = np.array([[-0.5, 0.5]])
    pts, wts = gauss_points_weights(n_gauss)
    ke = np.zeros((2, 2))

    for xi, w in zip(pts, wts):
        J = L / 2.0
        dN_dx = dN_dxi / J
        B = dN_dx

        if mat_type == "ELASTIC":
            E = params["E"]
            ke += B.T @ B * A * E * J * w

        elif mat_type == "NEOHOOKEAN":
            E = params["E"]
            nu = params.get("nu", 0.3)
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            # For small strain approximation: tangent stiffness ~ linear elastic
            ke += B.T @ B * A * (lam + 2 * mu) * J * w

        else:
            raise ValueError(f"Unsupported material type: {mat_type}")

    return ke

def elem_int_force(A: float, material: dict, xe: NDArray[float], ue: NDArray[float], n_gauss: int = 2) -> float: # type: ignore
    """Compute element internal axial force (positive in tension)
        Takes Inputs:
            A: element area
            E: Young's Modulus
            xe: element length
            ue: element displacement
            n_gauss: number of gauss points
        Returns:
            average internal force over element"""
    mat_type = material["type"].upper()
    params = material["parameters"]

    x1, x2 = xe[0, 0], xe[1, 0]
    L = x2 - x1
    if np.isclose(L, 0.0):
        raise ValueError("Zero-length element detected.")

    dN_dxi = np.array([[-0.5, 0.5]])
    pts, wts = gauss_points_weights(n_gauss)

    N_total = 0.0
    for xi, w in zip(pts, wts):
        J = L / 2.0
        dN_dx = dN_dxi / J
        B = dN_dx
        strain = float(B @ ue)

        if mat_type == "ELASTIC":
            stress = params["E"] * strain

        elif mat_type == "NEOHOOKEAN":
            E = params["E"]
            nu = params.get("nu", 0.3)
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            F = 1 + strain
            stress = (mu * (F**2 - 1) + lam * np.log(F)) / F  # 1D Neo-Hookean stress

        else:
            raise ValueError(f"Unsupported material type: {mat_type}")

        N_total += A * stress * J * w

    return float(N_total / L)


def elem_ext_force(A: float, q: float, he: float) -> NDArray[float]: # type: ignore
    """Compute equivalent nodal forces for a uniform distributed load q
        Takes Inputs:
            A: Element Area
            q: element distributed load
            he: element length
        Returns:
            external element force"""
    return q * he / 2 * np.ones(2)

# Global Assembly
def assem_glob_sys(coords, blocks, materials: dict, dof_per_node=1, n_gauss: int = 2):
    """Assemble global stiffness matrix from element stiffness contributions
        Takes Input:
            coords: element coordinates
            blocks: element blocks (see readme)
            materials: material parameters (see readme)
            dof_per_node: degrees of freedom per node
            n_gauss: number of gauss points
        Returns:
            K: global stiffness matrix"""
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node
    K = np.zeros((num_dof, num_dof))

    for block in blocks:
        A = block["element"]["properties"]["area"]
        mat = materials[block["material"]]
        E = mat["parameters"]["E"]

        for nodes in block["connect"]:
            eft = [glob_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            xe = coords[nodes]
            ke = elem_stiff(A, mat, xe, n_gauss)
            K[np.ix_(eft, eft)] += ke
    return K

def assem_ext_loads(coords, blocks, materials, dloads, block_elem_map, dof_per_node=1):
    """Assemble distributed loads into the global force vector
        Takes Inputs:
            coords: element coordinates
            blocks: element blocks (see readme)
            materials: material parameters (see readme)
            dloads: distributed loads
            block_elem_map: assign blocks to elements
            dof_per_node: degrees of freedom per node
        Returns:
            F: global force vector"""
    num_node = coords.shape[0]
    F = np.zeros(num_node * dof_per_node)

    for dload in dloads:
        dtype = dload["type"]
        direction = np.array(dload["direction"], dtype=float)
        sign = np.sign(direction[0])
        if sign == 0.0:
            raise ValueError(f"Invalid distributed load direction {direction}")

        for eid in dload["elements"]:
            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            nodes = block["connect"][local_index]
            xe = coords[nodes]
            he = xe[1, 0] - xe[0, 0]
            A = block["element"]["properties"]["area"]
            mat = materials[block["material"]]

            if dtype == "BX":
                q = dload["value"] * sign
            elif dtype == "GRAV":
                rho = mat["density"]
                q = rho * A * dload["value"] * sign
            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported")

            eft = [glob_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            Fe = elem_ext_force(A, q, he)
            F[eft] += Fe
    return F

def apply_bound_cond(K, F, bcs, dof_per_node=1):
    """Apply Dirichlet and Neumann BCs
        Takes Inputs:
            K: global stiffness matrix
            F: global force vector
            dof_per_node: degrees of freedom per node
        Returns:
            F: Adds boundary conditions to global force vector
            Array for prescribed degrees of freedom
            Array for prescribed boundary conditions at prescribed dofs"""
    num_dof = F.size
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = glob_dof(n, bc["local_dof"], dof_per_node)
                F[I] += bc["value"]

    prescribed_dofs, prescribed_vals = [], []
    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = glob_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(bc["value"])

    return F, np.array(prescribed_dofs), np.array(prescribed_vals)

# Solve System
def sys_solve(K, F, prescribed_dofs, prescribed_vals):
    """Solve Ku = F given prescribed DOFs
        Takes Inputs:
            K: global stiffness matrix
            F: global force vector
            prescribed_dofs: degrees of freedom with a prescribed boundary condition
            prescribed_vals: values of bc at prescribed dofs
        Returns:
            u: global displacement"""
    num_dof = F.size
    all_dofs = np.arange(num_dof)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, prescribed_dofs)]
    Ff = F[free_dofs] - Kfp.dot(prescribed_vals)

    uf = np.linalg.solve(Kff, Ff)
    u = np.zeros(num_dof)
    u[free_dofs] = uf
    u[prescribed_dofs] = prescribed_vals
    return u

def compute_int_forces(coords, blocks, materials, dofs):
    """Compute internal axial force for each element
        Takes Inputs:
            coords: nodal coordinates
            blocks: element blocks (see readme)
            materials: material parameters (see readme)
            dofs: element degrees of freedom
        Returns:
            global internal forces"""
    elem_forces = []
    for block in blocks:
        mat = materials[block["material"]]
        A = block["element"]["properties"]["area"]
        E = materials[block["material"]]["parameters"]["E"]
        for nodes in block["connect"]:
            xe = coords[nodes]
            ue = dofs[nodes]
            N = elem_int_force(A, mat, xe, ue)
            elem_forces.append(N)
    return np.array(elem_forces)

def glob_dof(node, local_dof, dof_per_node):
    """Global degree of freedom index
        Takes Inputs:
            node: global nodes
            local_dof: nodal degrees of freedom
            dof_per_node: degrees of freedom per node
        Returns:
            global degrees of freedom"""
    return node * dof_per_node + local_dof

# Put it all together
def first_fe_code(coords: NDArray[float], blocks: list[dict],bcs: list[dict],dloads: list[dict],materials: dict[str, Any],block_elem_map: dict[int, tuple[int, int]]) -> dict[str, Any]:  # type: ignore
    """
    Assemble and solve a 1D finite element system for an axial bar problem from module functions above
        Takes Inputs:
            coords: nodal coordinates
            blocks: element blocks (see readme)
            bcs: boundary conditions
            dloads: distributed loads (see readme)
            materials: material parameters (see readme)
            block_elem_map: assign blocks to elements 
        Returns:
            solution: returns global degrees of freedom, global stiffness matrix, and global force vector"""
    K = assem_glob_sys(coords, blocks, materials)
    F = assem_ext_loads(coords, blocks, materials, dloads, block_elem_map)
    F, prescribed_dofs, prescribed_vals = apply_bound_cond(K, F, bcs)
    dofs = sys_solve(K, F, prescribed_dofs, prescribed_vals)
    elem_forces = compute_int_forces(coords, blocks, materials, dofs)

    solution = {"dofs": dofs, "stiff": K, "force": F}

    return solution
