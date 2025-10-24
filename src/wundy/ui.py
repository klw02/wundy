import logging
from typing import IO
from typing import Any

import numpy as np
import yaml

from .schemas import DIRICHLET
from .schemas import NEUMANN
from .schemas import input_schema

logger = logging.getLogger(__name__)


def load(file: IO[Any]) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(file)
    return input_schema.validate(data)


def preprocess(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Preprocess and transform user input.

    Assumptions: User input was loaded and validated by ``load``

    """
    errors: int = 0

    inp = data["wundy"]

    preprocessed: dict[str, Any] = {}
    coords = preprocessed["coords"] = inp["coords"]
    connect = preprocessed["connect"] = inp["connect"]

    num_node, dof_per_node = coords.shape
    num_elem, node_per_elem = connect.shape

    # Put node sets in dictionary for easier look up
    nodesets = preprocessed.setdefault("nodesets", {})
    nodesets["all"] = list(range(num_node))
    for ns in inp.get("nset", []):
        if ns["name"] in nodesets:
            errors += 1
            logger.error(f"Duplicate node set {ns['name']}")
        else:
            nodesets[ns["name"]] = ns["nodes"]

    # Put element sets in dictionary for easier look up
    elsets = preprocessed.setdefault("element sets", {})
    elsets["all"] = list(range(num_elem))
    for es in inp.get("elset", []):
        if es["name"] in elsets:
            errors += 1
            logger.error(f"Duplicate element set {es['name']}")
        else:
            elsets[es["name"]] = es["elements"]

    # Put materials in dictionary for easier look up
    materials = preprocessed.setdefault("materials", {})
    for material in inp["material"]:
        if material["name"] in materials:
            errors += 1
            logger.error(f"Duplicate material {material['name']}")
        else:
            materials[material["name"]] = {
                "type": material["type"],
                "parameters": material["parameters"],
            }

    # Put element blocks in dictionary for easier look up
    blocks = preprocessed.setdefault("element blocks", {})
    for eb in inp["element block"]:
        if eb["name"] in blocks:
            errors += 1
            logger.error(f"Duplicate element block {eb['name']}")
            continue
        if eb["material"] not in materials:
            errors += 1
            logger.error(
                f"material {eb['material']!r} required by element block {eb['name']} not defined"
            )
            continue
        block = blocks.setdefault(eb["name"], {})
        block["element_properties"] = eb["element_properties"]
        block["material"] = eb["material"]
        block["element_type"] = eb["element_type"]
        if isinstance(eb["elements"], str):
            # elements given as set name
            if eb["elements"] not in elsets:
                errors += 1
                logger.error(
                    f"element set {eb['elements']!r} "
                    f"required by element eb {block['name']} not defined"
                )
                continue
            block["elements"] = elsets[eb["elements"]]
        else:
            block["elements"] = eb["elements"]

    # Convert boundary conditions to tags/vals that can be used by the assembler
    doftags = preprocessed["doftags"] = np.zeros((num_node, dof_per_node), dtype=int)
    dofvals = preprocessed["dofvals"] = np.zeros((num_node, dof_per_node), dtype=float)
    for boundary in inp["boundary"]:
        nodes: list[int] = []
        if "node" in boundary:
            nodes.append(boundary["node"])
        elif boundary["nset"] in nodesets:
            nodes.extend(nodesets[boundary["nset"]])
        else:
            errors += 1
            logger.error(f"nodeset {boundary['nset']} not defined")
            continue
        tag = DIRICHLET if boundary["type"] == "dirichlet" else NEUMANN
        dof = boundary["dof"]
        for node in nodes:
            doftags[node, dof] = tag
            dofvals[node, dof] = boundary["amplitude"]

    # Convert concentrated loads to tags/vals that can be used by the assembler
    for load in inp.get("cload", []):
        nodes: list[int] = []
        if "node" in load:
            nodes.append(load["node"])
        elif load["nset"] in nodesets:
            nodes.extend(nodesets[load["nset"]])
        else:
            errors += 1
            logger.error(f"nodeset {load['nset']} is not defined")
            continue
        dof = load["dof"]
        for node in nodes:
            # cload is a boundary condition of type 'neumann' with tag=0
            dofvals[node, dof] = load["amplitude"]

    # Process distributed load
    dload = preprocessed["dload"] = np.zeros((num_elem, dof_per_node), dtype=float)
    for load in inp.get("dload", []):
        elements: list[int] = []
        if "element" in load:
            elements.append(load["element"])
        elif load["elset"] in elsets:
            elements.extend(elsets[load["elset"]])
        else:
            errors += 1
            logger.error(f"element set {load['elset']} is not defined")
            continue
        dof = load["dof"]
        for element in elements:
            dload[element, dof] = load["amplitude"]

    # Check if all elements are assigned to an element block
    assigned: set[int] = set()
    for block in blocks.values():
        assigned.update(block["elements"])
    if unassigned := set(range(num_elem)).difference(assigned):
        errors += 1
        s = ", ".join(str(_) for _ in unassigned)
        logger.error(f"elements {s} are not assigned to an element block")

    if errors:
        raise ValueError("stopping due to previous errors")

    return preprocessed
