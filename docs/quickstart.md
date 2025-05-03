# Quick Start Guide

This guide will get you started with PydanticRDF quickly, showing basic usage patterns.

## Define Your First Model

To use PydanticRDF, you create classes that inherit from `BaseRdfModel` and define RDF mapping details:

```python
from rdflib import SDO
from pydantic_rdf import BaseRdfModel, WithPredicate
from pydantic import Annotated

# Define a model using Schema.org types
class Person(BaseRdfModel):
    # RDF type for this model (maps to rdf:type)
    rdf_type = SDO.Person
    
    # Default namespace for properties
    _rdf_namespace = SDO
    
    # Model fields
    name: str
    email: str
    job_title: Annotated[str, WithPredicate(SDO.jobTitle)] # Custom predicate
```

## Create and Serialize Instances

Once you have defined your model, you can create instances and serialize them to RDF:

```python
# Create an instance
person = Person(
    uri=SDO.Person_1,  # URI is a required field for all RDF models
    name="John Doe",
    email="john.doe@example.com",
    job_title="Software Engineer"
)

# Serialize to RDF graph
graph = person.model_dump_rdf()

# Print the graph as Turtle format
print(graph.serialize(format="turtle"))
```

The output will be an RDF graph with triples representing the model:

```
@prefix schema: <https://schema.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

schema:Person/1 a schema:Person ;
    schema:email "john.doe@example.com" ;
    schema:jobTitle "Software Engineer" ;
    schema:name "John Doe" .
```

## Deserialize from RDF

You can also deserialize RDF data back into model instances:

```python
# Parse an instance from the graph
loaded_person = Person.parse_graph(graph, SDO.Person_1)

# Access attributes
assert loaded_person.name == "John Doe"
assert loaded_person.email == "john.doe@example.com"
assert loaded_person.job_title == "Software Engineer"
```

## Working with Nested Models

PydanticRDF supports nested models and relationships:

```python
class PostalAddress(BaseRdfModel):
    rdf_type = SDO.PostalAddress
    _rdf_namespace = SDO
    
    streetAddress: str
    addressLocality: str

class PersonWithAddress(BaseRdfModel):
    rdf_type = SDO.Person
    _rdf_namespace = SDO
    
    name: str
    address: PostalAddress

# Create nested models
address = PostalAddress(uri=SDO.PostalAddress_1, streetAddress="123 Main St", addressLocality="Springfield")
person = PersonWithAddress(uri=SDO.Person_2, name="John Doe", address=address)

# Serialize to RDF
graph = person.model_dump_rdf()
```

## Working with Lists

PydanticRDF supports lists of items:

```python
class BlogPosting(BaseRdfModel):
    rdf_type = SDO.BlogPosting
    _rdf_namespace = SDO
    
    headline: str
    keywords: list[str]  # Will create multiple triples with the same predicate

# Create with a list
post = BlogPosting(
    uri=SDO.BlogPosting_1,
    headline="PydanticRDF Introduction",
    keywords=["RDF", "Pydantic", "Python"]
)

# Serialize to RDF
graph = post.model_dump_rdf()
```

## JSON Schema Generation

PydanticRDF supports generating valid JSON schemas for your RDF models:

```python
from pydantic import TypeAdapter

# Define your model as before
class Person(BaseRdfModel):
    rdf_type = SDO.Person
    _rdf_namespace = SDO
    
    name: str
    email: str

# Generate JSON schema
schema = TypeAdapter(Person).json_schema()

# URIRef fields will be properly represented as strings with URI format
# {
#   "properties": {
#     "uri": {
#       "type": "string",
#       "format": "uri",
#       "description": "The URI identifier for this RDF entity"
#     },
#     "name": {
#       "type": "string"
#     },
#     "email": {
#       "type": "string"
#     }
#   },
#   "required": ["uri", "name", "email"],
#   ...
# }
```

## Next Steps

Now that you have the basics, you can:

- Explore the [API reference](reference/pydantic_rdf/index.md) for detailed documentation
