# wundy
One-dimensional finite element program

---

## Installation

### 1. Clone the Repository

'''console
git clone git@github.com:klw02/wundy

### 2. Create a Virtual Environment

'''console
python3 -m venv venv
source venv/bin/activate

### 3. Install in Editable Mode

'''console
cd wundy
python3 -m pip install -e .

### 4. Running Tests
'''console
pytest

## User Input
wundy reads a model description from a YAML input file.
The supported input schemas are specified below:

### Nodes
Defines the coordinates and ID of each node:

nodes: [[id, x], [id, x], ....]

Example input:

nodes: [[1, 0.0], [2, 1.0], [3, 2.0]]

### Elements
Defines element ID and connectivity:

elements: [[element_id, start_node, end_node], ....]

Example input:

elements: [[1, 1, 2], [2, 2, 3]]

### Materials
Defines material type and associated properties
Example input:

materials:
  - type: elastic
    name: steel
    parameters:
      E: 210e9
      nu: 0.3
    density: 7800

Supported Fields:
  type: ELASTIC or NEOHOOK (NEOHOOK validation not yet implemented)
  parameters:
    E: Youngs Modulus (required)
    nu: Poisson's Ratio (optional, default = 0.0)
  density: mass density (optional, default - 0.0)

### Element Blocks
Groups material assignments and element properties
Example Input:

element blocks:
  - name: block-1
    material: steel
    elements: [1, 2]
    element:
      type: T1D1
      properties:
        area: 1.0

### Boundary Conditions
Assigns boundary conditions to prescribed node
Supports DIRICHLET boundary condition (prescribed displacement)
Supprts NEUMANN boundary condition (prescribed traction/force)
Example Input:

boundary conditions:
  - name: fix-left
    type: DIRICHLET
    dof: X
    nodes: [1]
    value: 0.0

  - name: right-force
    type: NEUMANN
    dof: X
    nodes: [3]
    value: 5.0

### Distributed Loads
Applies a constant load over prescribed elements
Example Input:

distributed loads:
  - name: distload
    type: BX
    direction: [1]
    value: 2.0
    elements: [1, 2]

Valid Types:
- BX: mechanical distributed load
- GRAV: gravitational body force

### Node Sets (Optional)
Example input:

node sets:
  - name: LEFT
    nodes: [1]

### Element Sets (Optional)
Example Input:

element sets:
  - name: CENTER
    elements [2]

### Concentrated Loads (Optional)
Equivalent to a NEUMANN boundary condition, but without specifying type
Example Input:

concentrated loads:
  - name: point-load
    dof: X
    nodes: [3]
    value: 5.0

## Full Example YAML Input
wundy:
  nodes: [[1, 0.0], [2, 1.0], [3, 2.0]]

  elements:
    - [1, 1, 2]
    - [2, 2, 3]

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
      elements: [1, 2]
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


