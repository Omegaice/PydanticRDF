import json
import logging
from collections.abc import MutableMapping, Sequence
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    Self,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from rdflib import RDF, Graph, Literal, URIRef

from pydantic_rdf.annotation import WithPredicate
from pydantic_rdf.exceptions import CircularReferenceError, UnsupportedFieldTypeError
from pydantic_rdf.types import IsDefinedNamespace, IsPrefixNamespace, TypeInfo

logger = logging.getLogger(__name__)


T = TypeVar("T", bound="BaseRdfModel")
M = TypeVar("M", bound="BaseRdfModel")  # For cls parameter annotations

# Sentinel object to detect circular references during parsing
_IN_PROGRESS: Final = object()


CacheKey: TypeAlias = tuple[type["BaseRdfModel"], URIRef]
RDFEntityCache: TypeAlias = MutableMapping[CacheKey, object]


class BaseRdfModel(BaseModel):
    """Base class for RDF-mappable Pydantic models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class variables for RDF mapping
    rdf_type: ClassVar[URIRef]
    _rdf_namespace: ClassVar[IsPrefixNamespace | IsDefinedNamespace]

    uri: URIRef = Field(description="The URI identifier for this RDF entity")

    # TYPE ANALYSIS HELPERS
    @classmethod
    def _get_field_predicate(cls: type[M], field_name: str, field: FieldInfo) -> URIRef:
        """Return the RDF predicate URI for a given field name and FieldInfo.

        Returns:
            The RDF predicate URIRef for the field.
        """
        if predicate := WithPredicate.extract(field):
            return predicate
        return cls._rdf_namespace[field_name]

    @staticmethod
    def _get_annotated_type(annotation: Any) -> Any | None:
        """Return the type wrapped by Annotated, or None if not Annotated.

        Args:
            annotation: The type annotation to inspect.

        Returns:
            The type wrapped by Annotated, or None if not Annotated.
        """
        if get_origin(annotation) is Annotated:
            return get_args(annotation)[0]
        return None

    @staticmethod
    def _get_union_type(annotation: Any) -> Any | None:
        """Return the first non-None type in a Union/Optional annotation, or None.

        Args:
            annotation: The type annotation to inspect.

        Returns:
            The first non-None type in the Union, or None if not a Union.
        """
        if get_origin(annotation) is Union:
            # Return the first non-None type (for Optional/Union)
            return next((arg for arg in get_args(annotation) if arg is not type(None)), None)
        return None

    @staticmethod
    def _get_sequence_type(annotation: Any) -> Any | None:
        """Return the item type if annotation is a Sequence (not str), else None.

        Args:
            annotation: The type annotation to inspect.

        Returns:
            The item type if annotation is a Sequence (not str), else None.
        """
        origin = get_origin(annotation)
        if isinstance(origin, type) and issubclass(origin, Sequence) and not issubclass(origin, str):
            return get_args(annotation)[0]
        return None

    @classmethod
    def _get_item_type(cls, annotation: Any) -> Any:
        """Recursively unwraps annotation to return the underlying item type.

        Args:
            annotation: The type annotation to unwrap.

        Returns:
            The underlying item type after unwrapping Annotated, Sequence, and Union.
        """
        for extractor in (cls._get_annotated_type, cls._get_sequence_type, cls._get_union_type):
            if (item_type := extractor(annotation)) is not None:
                return cls._get_item_type(item_type)
        return annotation

    @classmethod
    def _resolve_type_info(cls, annotation: Any) -> TypeInfo:
        """Return TypeInfo indicating if annotation is a list and its item type.

        Args:
            annotation: The type annotation to analyze.

        Returns:
            TypeInfo indicating whether the annotation is a list and its item type.
        """
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
        """Return the BaseRdfModel subclass from a type annotation, or None.

        Args:
            type_annotation: The type annotation to inspect.

        Returns:
            The BaseRdfModel subclass if present, else None.
        """
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
        """Convert an RDF value to a Python value or nested BaseRdfModel instance.

        Returns:
            The converted Python value or BaseRdfModel instance.

        Raises:
            CircularReferenceError: If a circular reference is detected during parsing.
        """
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
        """Extract and convert the value(s) for a field from the RDF graph.

        Returns:
            The extracted and converted value(s) for the field, or None if not present.

        Raises:
            UnsupportedFieldTypeError: If the field type is not supported for RDF parsing.
        """
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
        """Parse an RDF entity from the graph into a model instance.

        Uses a cache to prevent recursion and circular references.

        Args:
            _cache: Optional cache for already-parsed entities.

        Returns:
            An instance of the model corresponding to the RDF entity.

        Raises:
            CircularReferenceError: If a circular reference is detected during parsing.
            ValueError: If the URI does not have the expected RDF type.
            UnsupportedFieldTypeError: If a field type is not supported for RDF parsing.

        Example:
            ```python
            model = MyModel.parse_graph(graph, EX.some_uri)
            ```
        """
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

        # Update cache with the constructed instance
        cache[(cls, uri)] = instance

        return instance

    @classmethod
    def all_entities(cls: type[T], graph: Graph) -> list[T]:
        """Return all entities of this model's RDF type from the graph.

        Returns:
            A list of model instances for each entity of this RDF type in the graph.

        Raises:
            CircularReferenceError: If a circular reference is detected during parsing.
            ValueError: If any entity URI does not have the expected RDF type.
            UnsupportedFieldTypeError: If a field type is not supported for RDF parsing.

        Example:
            ```python
            entities = MyModel.all_entities(graph)
            ```
        """
        return [
            cls.parse_graph(graph, uri) for uri in graph.subjects(RDF.type, cls.rdf_type) if isinstance(uri, URIRef)
        ]

    # SERIALIZATION
    def model_dump_rdf(self: Self) -> Graph:
        """Serialize this model instance to an RDF graph.

        Returns:
            An RDFLib Graph representing this model instance.

        Example:
            ```python
            graph = instance.model_dump_rdf()
            ```
        """
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
