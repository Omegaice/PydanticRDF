from pydantic import TypeAdapter
from rdflib import Graph, Namespace, URIRef

from pydantic_rdf.model import BaseRdfModel
from pydantic_rdf.types import PydanticURIRef


def test_uriref_in_json_schema(EX: Namespace):
    """Test that URIRef fields are properly handled in JSON schema generation."""

    # Define test model
    class SimpleModel(BaseRdfModel):
        """A simple RDF model for testing schema generation."""

        rdf_type = EX.SimpleType
        _rdf_namespace = EX

        name: str
        description: str | None = None

    # Generate schema for the model
    schema = TypeAdapter(SimpleModel).json_schema()

    # Basic schema structure checks
    assert "properties" in schema
    assert "uri" in schema["properties"]

    # Verify URIRef is represented as a string with URI format
    uri_schema = schema["properties"]["uri"]
    assert uri_schema["type"] == "string"
    assert uri_schema["format"] == "uri"


def test_direct_uriref_handling():
    """Test direct handling of PydanticURIRef as a field type."""
    # Create a direct reference to the URIRef
    uri = URIRef("http://example.com/resource/1")

    # Convert to and validate as PydanticURIRef
    type_adapter = TypeAdapter(PydanticURIRef)

    # Test string conversion
    converted = type_adapter.validate_python("http://example.com/resource/2")
    assert isinstance(converted, URIRef)
    assert str(converted) == "http://example.com/resource/2"

    # Test direct URIRef validation
    validated = type_adapter.validate_python(uri)
    assert isinstance(validated, URIRef)
    assert validated == uri


def test_model_with_uriref_field_schema(graph: Graph, EX: Namespace):
    """Test that a model with URIRef fields can be properly serialized and deserialized."""

    class ResourceModel(BaseRdfModel):
        """Model with URIRef field for testing."""

        rdf_type = EX.Resource
        _rdf_namespace = EX

        name: str
        related_resource: PydanticURIRef | None = None

    # Create an instance with a URIRef
    related = URIRef("http://example.com/related")
    resource = ResourceModel(uri=EX.resource1, name="Test Resource", related_resource=related)

    # Add to graph
    graph += resource.model_dump_rdf()

    # Verify triples are in the graph using string comparison
    found_name = False
    found_related = False

    for s, p, o in graph:
        if str(s) == str(EX.resource1) and str(p) == str(EX.name) and str(o) == "Test Resource":
            found_name = True

        if str(s) == str(EX.resource1) and str(p) == str(EX.related_resource) and str(o) == str(related):
            found_related = True

    assert found_name, "Name triple not found"
    assert found_related, "Related resource triple not found"

    # Generate and verify schema
    schema = TypeAdapter(ResourceModel).json_schema()

    assert "related_resource" in schema["properties"]
    # Check that it's either a string or has anyOf with a string option
    related_schema = schema["properties"]["related_resource"]

    # Optional fields might use a different schema structure with anyOf
    if "type" in related_schema:
        assert related_schema["type"] == "string"
        assert related_schema.get("format") == "uri"
    elif "anyOf" in related_schema:
        # Find the string type option and check it
        string_options = [t for t in related_schema["anyOf"] if t.get("type") == "string"]
        assert string_options, "No string type option found in anyOf schema"
        assert any(t.get("format") == "uri" for t in string_options)
