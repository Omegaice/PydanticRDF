import json
import logging
from collections.abc import MutableMapping, Sequence
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    NamedTuple,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic.fields import FieldInfo
from rdflib import RDF, Graph, Literal, URIRef

from pydantic_rdf.annotation import WithPredicate

logger = logging.getLogger(__name__)


T = TypeVar("T", bound="BaseRdfModel")
M = TypeVar("M", bound="BaseRdfModel")  # For cls parameter annotations


class IsPrefixNamespace(Protocol):
    def __getitem__(self, key: str) -> URIRef: ...


class IsDefinedNamespace(Protocol):
    def __getitem__(self, name: str, default=None) -> URIRef: ...  # type: ignore


class IsNamespace(IsPrefixNamespace, IsDefinedNamespace, Protocol): ...  # type: ignore


# Type aliases for the RDF entity cache
CacheKey: TypeAlias = tuple[type["BaseRdfModel"], URIRef]
RDFEntityCache: TypeAlias = MutableMapping[CacheKey, Any]

# Sentinel object to detect circular references during parsing
_IN_PROGRESS: Final = object()


class TypeInfo(NamedTuple):
    is_list: bool
    item_type: Any


# Custom exceptions
class CircularReferenceError(Exception):
    """Raised when a circular reference is detected during RDF parsing."""

    def __init__(self, value: Any) -> None:
        message = f"Circular reference detected for {value}"
        super().__init__(message)


class UnsupportedFieldTypeError(Exception):
    """Raised when an unsupported field type is encountered during RDF parsing."""

    def __init__(self, field_type: Any, field_name: str) -> None:
        message = f"Unsupported field type: {field_type} for field {field_name}"
        super().__init__(message)


class BaseRdfModel(BaseModel):
    """Base class for RDF-mappable Pydantic models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class variables for RDF mapping
    rdf_type: ClassVar[URIRef]
    _rdf_namespace: ClassVar[IsNamespace]

    uri: URIRef = Field(description="The URI identifier for this RDF entity")
    _graph: Graph = PrivateAttr()

    # TYPE ANALYSIS HELPERS
    @classmethod
    def _get_field_predicate(cls: type[M], field_name: str, field: FieldInfo) -> URIRef:
        """Get RDF predicate URI for a field."""
        if predicate := WithPredicate.extract(field):
            return predicate
        return cls._rdf_namespace[field_name]

    @staticmethod
    def _get_annotated_type(annotation: Any) -> Any | None:
        if get_origin(annotation) is Annotated:
            return get_args(annotation)[0]
        return None

    @staticmethod
    def _get_union_type(annotation: Any) -> Any | None:
        if get_origin(annotation) is Union:
            # Return the first non-None type (for Optional/Union)
            return next((arg for arg in get_args(annotation) if arg is not type(None)), None)
        return None

    @staticmethod
    def _get_sequence_type(annotation: Any) -> Any | None:
        origin = get_origin(annotation)
        if isinstance(origin, type) and issubclass(origin, Sequence) and not issubclass(origin, str):
            return get_args(annotation)[0]
        return None

    @classmethod
    def _get_item_type(cls, annotation: Any) -> Any:
        """Given a list/sequence annotation, returns the item type, unwrapping Annotated, Union, etc."""
        for extractor in (cls._get_annotated_type, cls._get_sequence_type, cls._get_union_type):
            if (item_type := extractor(annotation)) is not None:
                return cls._get_item_type(item_type)
        return annotation

    @classmethod
    def _resolve_type_info(cls, annotation: Any) -> TypeInfo:
        """Analyzes a type annotation to determine if it represents a list type and extracts
        the underlying item type. Handles nested type constructs like Annotated, Union/Optional,
        and direct list/sequence types."""
        if (item_type := cls._get_sequence_type(annotation)) is not None:
            return TypeInfo(is_list=True, item_type=item_type)
        if (item_type := cls._get_annotated_type(annotation)) is not None:
            return cls._resolve_type_info(item_type)
        if (item_type := cls._get_union_type(annotation)) is not None:
            return TypeInfo(is_list=False, item_type=cls._get_item_type(item_type))
        return TypeInfo(is_list=False, item_type=annotation)

    # FIELD EXTRACTION AND CONVERSION
    @classmethod
    def _extract_model_type(cls, type_annotation: Any) -> type["BaseRdfModel"] | None:
        # Self reference
        if type_annotation is Self:
            return cls

        # Direct BaseRdfModel type
        if get_origin(type_annotation) is None:
            if (
                isinstance(type_annotation, type)
                and issubclass(type_annotation, BaseRdfModel)
                and type_annotation is not BaseRdfModel
            ):
                return type_annotation
            return None

        # Union/Optional types
        if (item_type := cls._get_union_type(type_annotation)) is not None:
            return cls._extract_model_type(item_type)

        return None

    @classmethod
    def _convert_rdf_value(
        cls: type[M],
        graph: Graph,
        value: Any,
        type_annotation: Any,
        cache: RDFEntityCache,
    ) -> Any:
        # Check if this is a nested BaseRdfModel
        if (model_type := cls._extract_model_type(type_annotation)) and isinstance(value, URIRef):
            # Handle nested BaseRdfModel instances with caching to prevent recursion
            if cached := cache.get((model_type, value)):
                # Check for circular references
                if cached is _IN_PROGRESS:
                    raise CircularReferenceError(value)
                return cached
            return model_type.parse_graph(graph, value, _cache=cache)

        # Convert literals to Python values
        if isinstance(value, Literal):
            python_value = value.toPython()
            # Handle JSON strings for dictionary fields
            origin = get_origin(type_annotation)
            if origin is dict and isinstance(python_value, str):
                try:
                    return json.loads(python_value)
                except json.JSONDecodeError:
                    pass  # If not valid JSON, return as is
            return python_value

        return value

    @classmethod
    def _extract_field_value(
        cls: type[M],
        graph: Graph,
        uri: URIRef,
        field_name: str,
        field: FieldInfo,
        cache: RDFEntityCache,
    ) -> Any | None:
        # Get all values for this predicate
        predicate = cls._get_field_predicate(field_name, field)
        values = list(graph.objects(uri, predicate))
        if not values:
            return None

        # Check if this is a list type
        type_info = cls._resolve_type_info(field.annotation)

        # Check for unsupported types
        if type_info.item_type is complex:
            raise UnsupportedFieldTypeError(type_info.item_type, field_name)

        # Process the values based on their type
        if type_info.is_list:
            return [cls._convert_rdf_value(graph, v, type_info.item_type, cache) for v in values]

        return cls._convert_rdf_value(graph, values[0], type_info.item_type, cache)

    # RDF PARSING
    @classmethod
    def parse_graph(cls: type[T], graph: Graph, uri: URIRef, _cache: RDFEntityCache | None = None) -> T:
        # Initialize cache if not provided
        cache: RDFEntityCache = {} if _cache is None else _cache

        # Return from cache if already constructed
        if cached := cache.get((cls, uri)):
            if cached is _IN_PROGRESS:
                raise CircularReferenceError(uri)
            return cast(T, cached)

        # Mark entry in cache as being built
        cache[(cls, uri)] = _IN_PROGRESS

        # Verify the entity has the correct RDF type
        if (uri, RDF.type, cls.rdf_type) not in graph:
            raise ValueError(f"URI {uri} does not have type {cls.rdf_type}")

        # Collect field data from the graph
        data: dict[str, Any] = {}
        for field_name, field in cls.model_fields.items():
            if field_name in BaseRdfModel.model_fields:
                continue
            value = cls._extract_field_value(graph, uri, field_name, field, cache)
            if value is not None:
                data[field_name] = value

        # Construct the instance with validation
        instance = cls.model_validate({"uri": uri, **data})

        # Set private attributes after construction
        private_attrs = {"_graph": graph}
        for k, v in private_attrs.items():
            setattr(instance, k, v)

        # Update cache with the constructed instance
        cache[(cls, uri)] = instance

        return instance

    @classmethod
    def all_entities(cls: type[T], graph: Graph) -> list[T]:
        """Get all entities of this model's type from the graph."""
        return [
            cls.parse_graph(graph, uri) for uri in graph.subjects(RDF.type, cls.rdf_type) if isinstance(uri, URIRef)
        ]

    # SERIALIZATION
    def model_dump_rdf(self: Self) -> Graph:
        """Serialize a model instance to an RDF graph."""
        graph = Graph()
        graph.add((self.uri, RDF.type, self.rdf_type))

        dumped = self.model_dump()

        for field_name, field in type(self).model_fields.items():
            if field_name == "uri":
                continue

            type_info = type(self)._resolve_type_info(field.annotation)

            # Use attribute value for BaseRdfModel fields, else use dumped value
            if (
                type_info.is_list
                and isinstance(type_info.item_type, type)
                and issubclass(type_info.item_type, BaseRdfModel)
            ) or (isinstance(type_info.item_type, type) and issubclass(type_info.item_type, BaseRdfModel)):
                value = getattr(self, field_name, None)
            else:
                value = dumped.get(field_name, None)

            if value is None:
                continue

            predicate = self._get_field_predicate(field_name, field)

            # Handle list fields
            if type_info.is_list and isinstance(value, list):
                if isinstance(type_info.item_type, type) and issubclass(type_info.item_type, BaseRdfModel):
                    for item in value:
                        if isinstance(item, BaseRdfModel):
                            graph.add((self.uri, predicate, item.uri))
                            graph += item.model_dump_rdf()
                    continue
                else:
                    # List of simple types
                    for item in value:
                        graph.add((self.uri, predicate, Literal(item)))
                    continue

            # Handle single BaseRdfModel
            if isinstance(type_info.item_type, type) and issubclass(type_info.item_type, BaseRdfModel):
                if isinstance(value, BaseRdfModel):
                    graph.add((self.uri, predicate, value.uri))
                    graph += value.model_dump_rdf()
            else:
                # Special handling for dict fields: serialize as JSON string
                origin = get_origin(type_info.item_type)
                if origin is dict and isinstance(value, dict):
                    graph.add((self.uri, predicate, Literal(json.dumps(value))))
                else:
                    graph.add((self.uri, predicate, Literal(value)))

        return graph
