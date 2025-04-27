#!/usr/bin/env python3
"""
Basic usage example for PydanticRDF (pydantic-rdf) using Schema.org types.

This example demonstrates:
1. Defining model classes with RDF mapping (Schema.org)
2. Creating and serializing instances to RDF
3. Deserializing RDF back to model instances
"""

from typing import Annotated

from pydantic import Field
from rdflib import SDO

from pydantic_rdf import BaseRdfModel, WithPredicate


# Define a simple model using Schema.org
class Person(BaseRdfModel):
    """A person model with RDF mapping (Schema.org)."""

    rdf_type = SDO.Person  # RDF type for this model
    _rdf_namespace = SDO  # Default namespace for properties

    name: str
    email: Annotated[str, WithPredicate(SDO.email)]
    jobTitle: Annotated[str, WithPredicate(SDO.jobTitle)]


# Define a nested model using Schema.org
class PostalAddress(BaseRdfModel):
    """A postal address model with RDF mapping (Schema.org)."""

    rdf_type = SDO.PostalAddress
    _rdf_namespace = SDO

    streetAddress: str
    addressLocality: str
    addressCountry: str = "Unknown"  # Field with default value


class PersonWithAddress(BaseRdfModel):
    """A person model with a nested address (Schema.org)."""

    rdf_type = SDO.Person
    _rdf_namespace = SDO

    name: str
    address: Annotated[PostalAddress, WithPredicate(SDO.address)]
    # List of strings (Schema.org: knowsAbout)
    knowsAbout: list[str] = Field(default_factory=list)


def main():
    # Create a simple Person instance
    person = Person(uri=SDO.Person_1, name="John Doe", email="john.doe@example.com", jobTitle="Software Engineer")

    # Serialize to RDF graph
    graph = person.model_dump_rdf()

    # Print the graph as Turtle
    print("Person RDF Graph:")
    print(graph.serialize(format="turtle"))
    print()

    # Deserialize back from the graph
    loaded_person = Person.parse_graph(graph, SDO.Person_1)
    print(f"Loaded Person: {loaded_person.name}, {loaded_person.email}, {loaded_person.jobTitle}")
    print()

    # Create nested models
    address = PostalAddress(
        uri=SDO.PostalAddress_1,
        streetAddress="123 Main St",
        addressLocality="Springfield",
    )

    person_with_address = PersonWithAddress(
        uri=SDO.Person_2, name="Jane Smith", address=address, knowsAbout=["RDF", "Pydantic", "Python"]
    )

    # Serialize to RDF graph
    graph2 = person_with_address.model_dump_rdf()

    # Print the graph as Turtle
    print("Person with Address RDF Graph:")
    print(graph2.serialize(format="turtle"))
    print()

    # Deserialize back from the graph
    loaded_person2 = PersonWithAddress.parse_graph(graph2, SDO.Person_2)
    print(f"Loaded Person: {loaded_person2.name}")
    print(f"Address: {loaded_person2.address.streetAddress}, {loaded_person2.address.addressLocality}")
    print(f"Knows About: {', '.join(loaded_person2.knowsAbout)}")


if __name__ == "__main__":
    main()
