import numpy as np
import wundy.first as wf

def test_elem_stiff_basic():
    A, E = 1.0, 200.0
    xe = np.array([[0.0], [2.0]])
    ke = wf.elem_stiff(A, E, xe)
    expected = A * E / 2 * np.array([[1, -1], [-1, 1]])
    assert np.allclose(ke, expected)

def test_elem_int_force_tension():
    A, E = 1.0, 100.0
    xe = np.array([[0.0], [2.0]])
    ue = np.array([0.0, 0.1])
    N = wf.elem_int_force(A, E, xe, ue)
    expected = A * E * (ue[1] - ue[0]) / 2
    assert np.isclose(N, expected)

def test_elem_ext_force():
    q, he = 5.0, 2.0
    A = 1.0
    Fe = wf.elem_ext_force(A, q, he)
    expected = q * he / 2 * np.ones(2)
    assert np.allclose(Fe, expected)

def test_global_assembly_small():
    coords = np.array([[0.0], [1.0], [2.0]])
    blocks = [{"element": {"properties": {"area": 1.0}}, "material": "MAT-1", "connect": np.array([[0, 1], [1, 2]])}]
    materials = {"MAT-1": {"parameters": {"E": 200.0}}}
    K = wf.assem_glob_sys(coords, blocks, materials)
    assert K.shape == (3, 3)
    assert np.isclose(K[0, 0], 200.0)
    assert np.isclose(K[-1, -1], 200.0)

def test_apply_dirichlet_bc():
    K = np.eye(3)
    F = np.zeros(3)
    bcs = [{"type": wf.DIRICHLET, "nodes": [0], "local_dof": 0, "value": 0.0}]
    F_mod, dofs, vals = wf.apply_bound_cond(K, F, bcs)
    assert np.allclose(F_mod, F)
    assert np.array_equal(dofs, [0])
    assert np.array_equal(vals, [0.0])

def test_solve_system_consistency():
    K = np.array([[2.0, -1.0], [-1.0, 2.0]])
    F = np.array([0.0, 1.0])
    prescribed_dofs = np.array([0])
    prescribed_vals = np.array([0.0])
    u = wf.sys_solve(K, F, prescribed_dofs, prescribed_vals)
    assert np.isclose(u[1], 0.5)
