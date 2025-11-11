import io

import numpy as np

import wundy
import wundy.first

def test_elem_stiff_basic():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
  distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    A, E = 1, 10
    ke = wundy.first.elem_stiff(inp["blocks"], inp["materials"], inp["coords"], n_gauss=2)
    expected = A * E / 2 * np.array([[1, -1], [-1, 1]])
    assert np.allclose(ke, expected)

def test_elem_int_force_tension():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
  distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)

    N = wundy.first.elem_int_force(inp["blocks"], inp["materials"], inp["coords"], ue)
    expected = A * E * (ue[1] - ue[0]) / 2
    assert np.isclose(N, expected)

def test_elem_ext_force():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
  distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)

    q, he = 2.0, 1.0
    A = 1.0
    Fe = wundy.first.elem_ext_force(inp["blocks"], inp["dload"], inp["block_elem_map"])
    expected = q * he / 2 * np.ones(2)
    assert np.allclose(Fe, expected)

def test_global_assembly_small():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
  distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)

    coords = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    blocks = [{"element": {"properties": {"area": 1.0}}, "material": "MAT-1", "connect": np.array([[0, 1], [1, 2]])}]
    materials = {"MAT-1": {"parameters": {"E": 10.0}}}
    K = wundy.first.assem_glob_sys(inp["coords"], inp["blocks"], inp["materials"], dof_per_node=1, n_gauss=2)
    assert K.shape == (5, 5)
    assert np.isclose(K[0, 0], 10.0)
    assert np.isclose(K[-1, -1], 10.0)

def test_apply_dirichlet_bc():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
  distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    
    K = np.eye(3)
    F = np.zeros(3)
    bcs = [{"type": wundy.first.DIRICHLET, "nodes": [0], "local_dof": 0, "value": 0.0}]
    F_mod, dofs, vals = wundy.first.apply_bound_cond(K, F, bcs)
    assert np.allclose(F_mod, F)
    assert np.array_equal(dofs, [0])
    assert np.array_equal(vals, [0.0])

def test_solve_system_consistency():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
  distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)

    K = np.array([[2.0, -1.0], [-1.0, 2.0]])
    F = np.array([0.0, 1.0])
    prescribed_dofs = np.array([0])
    prescribed_vals = np.array([0.0])
    u = wundy.first.sys_solve(K, F, prescribed_dofs, prescribed_vals)
    assert np.isclose(u[1], 0.5)
