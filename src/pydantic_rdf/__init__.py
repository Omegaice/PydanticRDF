"""PydanticRDF - Bridge between Pydantic V2 models and RDF graphs.

This library allows you to define data models with Pydantic's powerful validation
while working with semantic web technologies using rdflib. It enables seamless
serialization between Pydantic models and RDF graphs, and vice versa.

Example:
    ```python
    from rdflib import Namespace, URIRef
    from pydantic_rdf import BaseRdfModel, WithPredicate
    from pydantic import Field, Annotated

    # Define a namespace
    EX = Namespace("http://example.org/")

    # Define your model
    class Person(BaseRdfModel):
        rdf_type = EX.Person  # RDF type for this model
        _rdf_namespace = EX   # Default namespace for properties

        name: str
        age: int
        email: Annotated[str, WithPredicate(EX.emailAddress)]  # Custom predicate
    ```
"""

from pydantic_rdf.annotation import WithDataType, WithPredicate
from pydantic_rdf.model import BaseRdfModel

__all__ = ["BaseRdfModel", "WithDataType", "WithPredicate"]
