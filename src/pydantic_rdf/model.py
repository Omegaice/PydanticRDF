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


# Type alias for the RDF entity cache
CacheKey = tuple[type["BaseRdfModel"], URIRef]
RDFEntityCache = MutableMapping[CacheKey, Any]

# Sentinel object to detect circular references during parsing
_IN_PROGRESS: Final = object()


class TypeInfo(NamedTuple):
    is_list: bool
    item_type: Any


class BaseRdfModel(BaseModel):
    """Base class for RDF-mappable Pydantic models"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class variables for RDF mapping
    rdf_type: ClassVar[URIRef]
    _rdf_namespace: ClassVar[IsNamespace]

    uri: URIRef = Field(description="The URI identifier for this RDF entity")
    _graph: Graph = PrivateAttr()

    @classmethod
    def _get_field_predicate(cls: type[M], field_name: str, field: FieldInfo) -> URIRef:
        """Resolves the RDF predicate URI for a field by checking for WithPredicate annotation
        or falling back to the default namespace + field name pattern."""
        if predicate := WithPredicate.extract(field):
            return predicate
        return cls._rdf_namespace[field_name]

    @classmethod
    def _is_sequence_type(cls, origin: Any) -> bool:
        """Determines if a type origin represents a sequence type (excluding strings)."""
        return isinstance(origin, type) and issubclass(origin, Sequence) and not issubclass(origin, str)

    @classmethod
    def _resolve_union_type(cls, annotation: Any) -> Any:
        """Resolves a union type to its first non-None member type."""
        for arg in get_args(annotation):
            if arg is not type(None):
                return cls._resolve_type_info(arg)
        return annotation

    @classmethod
    def _resolve_type_info(cls, annotation: Any) -> TypeInfo:
        """Analyzes a type annotation to determine if it represents a list type and extracts
        the underlying item type. Handles nested type constructs like Annotated, Union/Optional,
        and direct list/sequence types."""
        origin = get_origin(annotation)

        # Handle Annotated types first
        if origin is Annotated:
            args = get_args(annotation)
            return cls._resolve_type_info(args[0])

        # Handle direct list/sequence types
        if cls._is_sequence_type(origin):
            return TypeInfo(is_list=True, item_type=get_args(annotation)[0])

        # Handle Union types (including Python 3.11's X | Y syntax)
        if origin is Union:
            return TypeInfo(is_list=False, item_type=cls._resolve_union_type(annotation))

        # Default case: not a list type
        return TypeInfo(is_list=False, item_type=annotation)

    @classmethod
    def _extract_BaseRdfModel_type(cls, type_annotation: Any) -> type["BaseRdfModel"] | None:
        """Examines a type annotation to find any BaseRdfModel subclass types. Handles direct types,
        Self references, and union types. Returns None if no BaseRdfModel type is found."""
        # Self reference
        if type_annotation is Self:
            return cls

        origin = get_origin(type_annotation)

        # Direct BaseRdfModel type
        if origin is None:
            if (
                isinstance(type_annotation, type)
                and issubclass(type_annotation, BaseRdfModel)
                and type_annotation is not BaseRdfModel
            ):
                return type_annotation
            return None

        # Union/Optional types
        if origin is Union:
            for arg in get_args(type_annotation):
                if arg is not type(None) and (rdf_type := cls._extract_BaseRdfModel_type(arg)):
                    return rdf_type

        return None

    @classmethod
    def _convert_rdf_value(
        cls: type[M],
        graph: Graph,
        value: Any,
        type_annotation: Any,
        cache: RDFEntityCache,
    ) -> Any:
        """Converts an RDF value to its corresponding Python type. Handles nested BaseRdfModel instances
        with caching to prevent recursion, converts RDF literals to Python values, and passes through
        other types unchanged."""
        # Check if this is a nested BaseRdfModel
        if (BaseRdfModel_type := cls._extract_BaseRdfModel_type(type_annotation)) and isinstance(value, URIRef):
            # Check for circular references
            if cache.get((BaseRdfModel_type, value)) is _IN_PROGRESS:
                raise RecursionError(f"Circular reference detected for {value}")

            # Handle nested BaseRdfModel instances with caching to prevent recursion
            if cached := cache.get((BaseRdfModel_type, value)):
                return cached
            return BaseRdfModel_type.parse_graph(graph, value, _cache=cache)

        # Convert literals to Python values
        if isinstance(value, Literal):
            python_value = value.toPython()

            # Handle JSON strings for dictionary fields
            origin = get_origin(type_annotation)
            if origin is dict and isinstance(python_value, str):
                try:
                    import json

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
        """Extracts a field's value from the RDF graph by finding all matching predicates and converting
        the values to the appropriate Python types. Handles both single values and lists, with special
        processing for nested BaseRdfModel instances."""
        # Get all values for this predicate
        predicate = cls._get_field_predicate(field_name, field)
        if not (values := list(graph.objects(uri, predicate))):
            return None

        # Check if this is a list type
        type_info = cls._resolve_type_info(field.annotation)

        # Check for unsupported types
        if type_info.item_type is complex:
            raise TypeError(f"Unsupported field type: {type_info.item_type} for field {field_name}")

        # Process the values based on their type
        if type_info.is_list:
            return [cls._convert_rdf_value(graph, v, type_info.item_type, cache) for v in values]

        return cls._convert_rdf_value(graph, values[0], type_info.item_type, cache)

    @classmethod
    def parse_graph(cls: type[T], graph: Graph, uri: URIRef, _cache: RDFEntityCache | None = None) -> T:
        """Creates a model instance by parsing data from an RDF graph. Maps RDF predicates to model fields,
        handles nested models with circular reference detection, and validates the resulting instance
        through Pydantic. The _cache parameter enables shared caching across recursive calls."""
        # Initialize cache if not provided
        cache: RDFEntityCache = {} if _cache is None else _cache

        # Return from cache if already constructed
        if cached := cache.get((cls, uri)):
            if cached is _IN_PROGRESS:
                raise RecursionError(f"Circular reference detected for {uri}")
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

            if value := cls._extract_field_value(graph, uri, field_name, field, cache):
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
        """Retrieves all entities of this model's type from the graph by finding all subjects
        with matching rdf_type. Filters out non-URIRef subjects and returns parsed instances."""
        return [
            cls.parse_graph(graph, uri) for uri in graph.subjects(RDF.type, cls.rdf_type) if isinstance(uri, URIRef)
        ]

    # Serialization
    def model_dump_rdf(self: Self) -> Graph:
        """Serializes a model instance to an RDF graph. Handles only simple string fields for this test."""
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
            if value is not None:
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
