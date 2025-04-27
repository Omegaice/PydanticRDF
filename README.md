<div align="center">

# PydanticRDF

[<img alt="github" src="https://img.shields.io/badge/github-Omegaice/pydantic--rdf-8da0cb?style=for-the-badge&logo=github" height="20">](https://github.com/Omegaice/pydantic-rdf)
[<img alt="PyPI" src="https://img.shields.io/pypi/v/pydantic-rdf?style=for-the-badge&color=1E88E5&logo=python" height="20">](https://pypi.org/project/pydantic-rdf)
[<img alt="Python" src="https://img.shields.io/pypi/pyversions/pydantic-rdf?style=for-the-badge&logo=python" height="20">](https://pypi.org/project/pydantic-rdf)
[<img alt="docs" src="https://img.shields.io/badge/docs-pydantic--rdf-blue?style=for-the-badge&logo=readthedocs" height="20">](https://omegaice.github.io/pydantic-rdf)
[<img alt="license" src="https://img.shields.io/github/license/Omegaice/pydantic-rdf?style=for-the-badge&color=green" height="20">](https://github.com/Omegaice/pydantic-rdf/blob/master/LICENSE)

</div>

A Python library that bridges Pydantic V2 models and RDF graphs, enabling seamless bidirectional conversion between typed data models and semantic web data. PydanticRDF combines Pydantic's powerful validation with RDFLib's graph capabilities to simplify working with linked data.

## Features

- ✅ **Type Safety**: Define data models with Pydantic V2 and map them to RDF graphs
- 🔄 **Serialization**: Convert Pydantic models to RDF triples with customizable predicates
- 📥 **Deserialization**: Parse RDF data into validated Pydantic models
- 🧩 **Complex Structures**: Support for nested models, lists, and circular references

## Installation

```bash
pip install pydantic-rdf
```

## Quick Example

```python
from rdflib import Namespace
from pydantic_rdf import BaseRdfModel, WithPredicate
from pydantic import Annotated

# Define a model
EX = Namespace("http://example.org/")

class Person(BaseRdfModel):
    rdf_type = EX.Person
    _rdf_namespace = EX
    
    name: str
    age: int
    email: Annotated[str, WithPredicate(EX.emailAddress)]

# Create and serialize
person = Person(uri=EX.person1, name="John Doe", age=30, email="john@example.com")
graph = person.model_dump_rdf()

# The resulting RDF graph looks like this:
# @prefix ex: <http://example.org/> .
# @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
#
# ex:person1 a ex:Person ;
#     ex:age 30 ;
#     ex:emailAddress "john@example.com" ;
#     ex:name "John Doe" .

# Deserialize
loaded_person = Person.parse_graph(graph, EX.person1)
```

## Requirements

- Python 3.11+
- pydantic >= 2.11.3
- rdflib >= 7.1.4

## Documentation

For complete documentation, visit [https://omegaice.github.io/pydantic-rdf/](https://omegaice.github.io/pydantic-rdf/)

## License

MIT