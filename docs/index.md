# PydanticRDF Documentation

Welcome to the PydanticRDF documentation! This library bridges Pydantic V2 models and RDF graphs, allowing seamless serialization and deserialization between them.

## Table of Contents

- [Quick Start](quickstart.md)
- [API Reference](reference/pydantic_rdf/index.md)

## Overview

PydanticRDF enables you to define your data models with Pydantic's powerful validation while working with semantic web technologies using rdflib. It makes it easy to:

- Map Pydantic models to RDF types
- Serialize model instances to RDF graphs
- Deserialize RDF data into typed, validated Pydantic models
- Handle nested models, circular references, and complex types

The library is designed to be simple and intuitive, following Pydantic's design principles.

## Key Features

- **Type Safety**: Full type checking and validation powered by Pydantic
- **Simple API**: Intuitive interfaces for serialization and deserialization
- **Seamless Integration**: Works with existing Pydantic and RDF workflows
- **Flexible Mapping**: Custom predicate and datatype annotations
- **Comprehensive Support**: Handles nested models, lists, and circular references
- **JSON Schema**: Generate valid JSON schemas with proper URI field handling

## Example

```python
from rdflib import SDO
from pydantic_rdf import BaseRdfModel, WithPredicate
from typing import Annotated

# Define a model using Schema.org types
class Person(BaseRdfModel):
    rdf_type = SDO.Person
    _rdf_namespace = SDO
    
    name: str
    email: str
    job_title: Annotated[str, WithPredicate(SDO.jobTitle)]

# Create an instance
person = Person(
    uri=SDO.Person_1,
    name="John Doe",
    email="john.doe@example.com",
    job_title="Software Engineer"
)

# Serialize to RDF
graph = person.model_dump_rdf()

# Deserialize from RDF
loaded_person = Person.parse_graph(graph, SDO.Person_1)
```

For more detailed examples and guides, check out the rest of the documentation!
