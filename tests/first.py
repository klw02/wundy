import io

import numpy as np

import wundy
import wundy.first


def test_first_1():
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
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    assert np.allclose(dofs, [0, 0.2, 0.4, 0.6, 0.8])
    assert np.allclose(F, [0, 0, 0, 0, 2])
    assert np.allclose(
        K,
        [
            [10, -10, 0, 0, 0],
            [-10, 20, -10, 0, 0],
            [0, -10, 20, -10, 0],
            [0, 0, -10, 20, -10],
            [0, 0, 0, -10, 10],
        ],
    )


def test_first_2():
    """Test 1D bar with uniform distributed load (BX type)."""
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2]]
  elements: [[1, 1, 2], [2, 2, 3]]
  boundary conditions:
  - name: fix-left
    dof: x
    nodes: [1]
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
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
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    u = soln["dofs"]
    F = soln["force"]

    # Check displacement pattern (should increase quadratically)
    assert np.all(u >= 0)
    assert u[-1] > u[-2] > u[0]

    # Check total equivalent nodal load equals qL
    total_load = F.sum()
    assert np.isclose(total_load, 2 * 2.0, rtol=1e-6)

