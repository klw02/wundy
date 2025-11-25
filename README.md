# wundy

wundy is a one-dimensional finite element analysis program. It builds and solves linear elastic problems using YAML-based input that describes the mesh, material properties, loading, and boundary conditions. The tool is designed for lightweight experimentation and validation of 1D bar and truss-style models.

---

## Installation

Install wundy directly from GitHub in a virtual environment:

1. **Clone the repository**

   ```bash
   git clone git@github.com:klw02/wundy
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install wundy in editable mode**

   ```bash
   cd wundy
   python3 -m pip install -e .
   ```

4. **(Optional) Run tests**

   ```bash
   pytest
   ```

## User Input

wundy reads a YAML file with a top-level `wundy` key. Each subsection below defines the expected structure and optional fields.

- **nodes** – Coordinates keyed by ID: `nodes: [[id, x], ...]`.
- **elements** – Element connectivity: `elements: [[element_id, start_node, end_node], ...]`.
- **materials** – Material definitions. Supported `type` values include `elastic` (and `NEOHOOK`, though Neohook validation is not yet implemented). Parameters include Young's modulus `E` (required), Poisson's ratio `nu` (optional, defaults to 0.0), and `density` (optional, defaults to 0.0).
- **element blocks** – Groups elements and assigns material plus element properties. Each block includes a `material`, `elements`, and an `element` section with `type` (e.g., `T1D1`) and `properties` such as `area`.
- **boundary conditions** – Supports `DIRICHLET` (prescribed displacement) and `NEUMANN` (prescribed traction/force). Each condition lists `dof`, `nodes`, and `value`.
- **distributed loads** – Constant loads over elements. Valid `type` values are `BX` for mechanical distributed load and `GRAV` for gravitational body force. Provide `direction`, `value`, and target `elements`.
- **node sets** *(optional)* – Named groups of nodes: `node sets: [{name, nodes}]`.
- **element sets** *(optional)* – Named groups of elements: `element sets: [{name, elements}]`.
- **concentrated loads** *(optional)* – Equivalent to `NEUMANN` boundary conditions but without specifying `type`; include `dof`, `nodes`, and `value`.

## Full Example YAML Input

```yaml
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
```
