from schema import SchemaError # type: ignore
from wundy.schemas import material_schema, validate_material_parameters


def test_elastic_material_validates():
    """Ensure the default ELASTIC material passes schema validation."""
    material = {
        "type": "elastic",
        "name": "steel",
        "parameters": {"E": 200000.0, "nu": 0.3},
        "density": 7850.0,
    }

    # Should pass without raising SchemaError
    result = material_schema.validate(material)
    assert result["type"] == "ELASTIC"
    assert "parameters" in result


def test_neohookean_material_validates():
    """Ensure the NEOHOOKEAN material passes schema validation."""
    material = {
        "type": "neoHookean",
        "name": "rubber",
        "parameters": {"E": 10.0, "nu": 0.45},
        "density": 1100.0,
    }

    # Should pass without raising SchemaError
    result = material_schema.validate(material)
    assert result["type"] == "NEOHOOKEAN"
    assert "parameters" in result


def test_invalid_material_type_fails():
    """Ensure unsupported material types raise ValueError or SchemaError."""
    material = {
        "type": "viscoelastic",
        "name": "test",
        "parameters": {"E": 10.0},
    }

    try:
        material_schema.validate(material)
        # If no error is raised, the test should fail
        assert False, "Expected ValueError or SchemaError but none was raised"
    except (ValueError, SchemaError):
        # Expected outcome → test passes
        pass



def test_missing_E_parameter_fails():
    """Ensure materials missing 'E' fail validation."""
    material = {
        "type": "neoHookean",
        "name": "rubber",
        "parameters": {"nu": 0.45},
    }

    try:
        validate_material_parameters(material)
        assert False, "Expected ValueError or SchemaError but none was raised"
    except (ValueError, SchemaError):
        # Expected outcome → test passes
        pass