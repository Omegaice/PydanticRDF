#!/usr/bin/env python3
"""
JSON Schema generation example for PydanticRDF (pydantic-rdf).

This example demonstrates:
1. Generating JSON schemas from RDF models
2. How URIRef fields are properly represented in the schema
3. Working with schema validation
"""

import json
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError
from rdflib import Namespace, URIRef

from pydantic_rdf import BaseRdfModel, WithPredicate
from pydantic_rdf.types import PydanticURIRef

# Define a namespace
EX = Namespace("http://example.org/")


# Define a simple RDF model
class Person(BaseRdfModel):
    """A person model for demonstrating schema generation."""

    rdf_type = EX.Person
    _rdf_namespace = EX

    name: str
    email: str
    age: int
    website: PydanticURIRef | None = None


# Define a more complex model with nested structure
class Address(BaseRdfModel):
    """An address model for demonstrating nested schema generation."""

    rdf_type = EX.Address
    _rdf_namespace = EX

    street: str
    city: str
    postal_code: str
    country: str


class ContactInfo(BaseRdfModel):
    """A contact info model with custom URIRef field."""

    rdf_type = EX.ContactInfo
    _rdf_namespace = EX

    email: str
    phone: str
    homepage: PydanticURIRef | None = None


class Organization(BaseRdfModel):
    """An organization model with nested models and URIRef fields."""

    rdf_type = EX.Organization
    _rdf_namespace = EX

    name: str
    address: Annotated[Address, WithPredicate(EX.hasAddress)]
    contact: Annotated[ContactInfo, WithPredicate(EX.hasContact)]
    related_orgs: list[PydanticURIRef] = Field(default_factory=list)


def main():
    # Generate JSON schema for Person model
    person_schema = TypeAdapter(Person).json_schema()

    # Pretty print the schema
    print("Person JSON Schema:")
    print(json.dumps(person_schema, indent=2))
    print()

    # Examine URI-related fields
    uri_field = person_schema["properties"]["uri"]
    print("URI Field Schema:")
    print(json.dumps(uri_field, indent=2))
    print()

    website_field = person_schema["properties"]["website"]
    print("Website Field Schema (Optional URIRef):")
    print(json.dumps(website_field, indent=2))
    print()

    # Generate schema for complex model with nested structures
    org_schema = TypeAdapter(Organization).json_schema()

    print("Organization JSON Schema (Summary):")
    print(f"Properties: {list(org_schema['properties'].keys())}")
    print(f"Required: {org_schema.get('required', [])}")
    print()

    # Demonstrate validation with the schema
    # Valid data
    valid_person_data = {
        "uri": "http://example.org/person/1",
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "website": "https://johndoe.com",
    }

    try:
        person = Person.model_validate(valid_person_data)
        print(f"Valid person created: {person.name}, {person.email}")
        print(f"URIRef fields: uri={person.uri}, website={person.website}")
        # Check if the fields are properly converted to URIRef instances
        is_uri_uriref = isinstance(person.uri, URIRef)
        is_website_uriref = isinstance(person.website, URIRef)
        print(f"Are URIRef instances? uri: {is_uri_uriref}, website: {is_website_uriref}")
        print()
    except ValidationError as e:
        print(f"Validation error: {e}")

    # Invalid data (wrong types)
    invalid_person_data = {
        "uri": "http://example.org/person/2",
        "name": "Jane Smith",
        "email": "jane@example.com",
        "age": "thirty",  # Should be an integer
        "website": 12345,  # Should be a URI string
    }

    try:
        Person.model_validate(invalid_person_data)
        print("This should not happen - validation should fail")
    except ValidationError as e:
        print("Expected validation error:")
        print(e)


if __name__ == "__main__":
    main()
