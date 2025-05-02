from typing import Annotated, Any, NamedTuple, Protocol

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from rdflib import URIRef


class IsPrefixNamespace(Protocol):
    def __getitem__(self, key: str) -> URIRef: ...


class IsDefinedNamespace(Protocol):
    def __getitem__(self, name: str, default: Any = None) -> URIRef: ...


class TypeInfo(NamedTuple):
    """Type information for a field: whether it is a list and its item type."""

    is_list: bool
    item_type: object


class URIRefHandler:
    """Handler for URIRef type to work with Pydantic schema generation."""

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler) -> CoreSchema:
        """Generate a core schema for URIRef.

        Returns a core schema that validates URIRefs and strings,
        converting strings to URIRefs automatically.
        """

        def validate_from_str(value: str) -> URIRef:
            return URIRef(value)

        from_str_schema = core_schema.chain_schema([
            core_schema.str_schema(),
            core_schema.no_info_plain_validator_function(validate_from_str),
        ])

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(URIRef),
                from_str_schema,
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda instance: str(instance)),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: CoreSchema, _handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Generate a JSON schema for URIRef.

        Returns a JSON schema that represents URIRef as a string with URI format.
        """
        return {"type": "string", "format": "uri"}


# Create an annotated type for URIRef
PydanticURIRef = Annotated[URIRef, URIRefHandler]
