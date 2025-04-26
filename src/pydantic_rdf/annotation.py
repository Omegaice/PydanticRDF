from dataclasses import dataclass

from rdflib import URIRef


@dataclass
class WithPredicate:
    """Annotation to specify a custom RDF predicate for a field."""

    predicate: URIRef

    @classmethod
    def extract(cls, field) -> URIRef | None:  # type: ignore
        """Extract from field annotation if present."""
        for meta in getattr(field, "metadata", []):
            if isinstance(meta, WithPredicate):
                return meta.predicate
        return None


@dataclass
class WithDataType:
    """Annotation to specify a custom RDF datatype for a field."""

    data_type: URIRef

    @classmethod
    def extract(cls, field) -> URIRef | None:  # type: ignore
        """Extract from field annotation if present."""
        for meta in getattr(field, "metadata", []):
            if isinstance(meta, WithDataType):
                return meta.data_type
        return None
