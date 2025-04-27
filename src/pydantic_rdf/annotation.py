from dataclasses import dataclass

from rdflib import URIRef


@dataclass
class WithPredicate:
    """Annotation to specify a custom RDF predicate for a field.

    This annotation allows you to define a specific RDF predicate to use when serializing
    a model field to RDF, instead of using the default predicate generated from the field name.

    Args:
        predicate: The RDF predicate URI to use for this field

    Example:
        ```python
        from typing import Annotated
        from pydantic_rdf import BaseRdfModel, WithPredicate
        from rdflib import Namespace

        EX = Namespace("http://example.org/")

        class Person(BaseRdfModel):
            # This will use the EX.emailAddress predicate instead of the default EX.email
            email: Annotated[str, WithPredicate(EX.emailAddress)]
        ```
    """

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
    """Annotation to specify a custom RDF datatype for a field.

    This annotation allows you to define a specific RDF datatype to use when serializing
    a model field to RDF literals, instead of using the default datatype inference.

    Args:
        data_type: The RDF datatype URI to use for this field

    Example:
        ```python
        from typing import Annotated
        from pydantic_rdf import BaseRdfModel, WithDataType
        from rdflib.namespace import XSD

        class Product(BaseRdfModel):
            # This will use xsd:decimal datatype instead of the default
            price: Annotated[float, WithDataType(XSD.decimal)]
        ```
    """

    data_type: URIRef

    @classmethod
    def extract(cls, field) -> URIRef | None:  # type: ignore
        """Extract from field annotation if present."""
        for meta in getattr(field, "metadata", []):
            if isinstance(meta, WithDataType):
                return meta.data_type
        return None
