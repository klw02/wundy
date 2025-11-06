# wundy
One dimension finite element program

## Install

### Clone repository

```console
git clone git@github.com:klw02/wundy
```

### Create virtual environment

```console
python3 -m venv venv
source activate venv/bin/activate
```

### Install in editable mode

```console
cd wundy
python3 -m pip install -e .
```

## Test

In the `wundy` directory, execute

```console
pytest
```
## 🧩 User Input Format

WUNDY reads model definitions from a **YAML input file** that specifies the geometry, materials, elements, boundary conditions, and loads.

All input files must begin with the top-level key:

```yaml
wundy:

Below are the supported sections and their required formats

nodes: [[id, x], [id, x], ...]

elements: [[element_id, start_node, end_node], ...]

materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
    density: 1.0

element blocks:
  - name: block-1
    material: mat-1
    elements: all
    element:
      type: T1D1
      properties:
        area: 1.0
    
boundary conditions:
  - name: fix-left
    type: DIRICHLET
    dof: X
    nodes: [1]
    value: 0.0

  - name: load-right
    type: NEUMANN
    dof: X
    nodes: [3]
    value: 5.0

distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]

node sets:
  - name: LEFT
    nodes: [1, 2]

element sets:
  - name: CENTER
    elements: [2]

concentrated loads:
  - name: point-load
    dof: X
    nodes: [3]
    value: 5.0

Full Example Input

wundy:
  nodes: [[1, 0.0], [2, 1.0], [3, 2.0]]
  elements: [[1, 1, 2], [2, 2, 3]]

  materials:
    - type: elastic
      name: mat-1
      parameters:
        E: 10.0
        nu: 0.3
      density: 1.0

  element blocks:
    - name: block-1
      material: mat-1
      elements: all
      element:
        type: T1D1
        properties:
          area: 1.0

  boundary conditions:
    - name: fix-left
      type: DIRICHLET
      dof: X
      nodes: [1]
      value: 0.0

  distributed loads:
    - name: distload
      type: BX
      direction: [1]
      value: 2.0
      elements: [1, 2]
     
     
