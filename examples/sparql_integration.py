#!/usr/bin/env python3
"""
SPARQL integration example for PydanticRDF (pydantic-rdf).

This example demonstrates:
1. Loading RDF data from an external source
2. Running SPARQL queries on the data
3. Converting query results to Pydantic models
"""

from typing import Annotated

from pydantic import Field
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import FOAF, RDF
from rdflib.query import ResultRow

from pydantic_rdf import BaseRdfModel, WithPredicate

# Define our namespaces
EX = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")


# Define our models
class Person(BaseRdfModel):
    """A person model with RDF mapping."""

    rdf_type = FOAF.Person
    _rdf_namespace = FOAF

    name: str
    knows: list["Person"] = Field(default_factory=list)
    age: int | None = None

    @classmethod
    def from_sparql_result(cls, graph: Graph, binding):
        """Create an instance from a SPARQL query result binding."""
        person_uri = binding.get("person")
        if not person_uri or not isinstance(person_uri, URIRef):
            raise ValueError("Invalid person URI in SPARQL binding")

        # Use the standard parse_graph method to create the instance
        return cls.parse_graph(graph, person_uri)


class Book(BaseRdfModel):
    """A book model with RDF mapping."""

    rdf_type = SCHEMA.Book
    _rdf_namespace = SCHEMA

    title: str
    author: Annotated[Person, WithPredicate(SCHEMA.author)]
    year: int | None = None
    isbn: str | None = None


def main():
    # Create a sample RDF graph
    g = Graph()

    # Add some triples about people
    g.add((EX.person1, RDF.type, FOAF.Person))
    g.add((EX.person1, FOAF.name, Literal("Alice Smith")))
    g.add((EX.person1, FOAF.age, Literal(35)))

    g.add((EX.person2, RDF.type, FOAF.Person))
    g.add((EX.person2, FOAF.name, Literal("Bob Jones")))
    g.add((EX.person2, FOAF.age, Literal(42)))

    g.add((EX.person3, RDF.type, FOAF.Person))
    g.add((EX.person3, FOAF.name, Literal("Charlie Miller")))

    # Add relationships between people
    g.add((EX.person1, FOAF.knows, EX.person2))
    g.add((EX.person2, FOAF.knows, EX.person3))
    g.add((EX.person3, FOAF.knows, EX.person1))

    # Add some books
    g.add((EX.book1, RDF.type, SCHEMA.Book))
    g.add((EX.book1, SCHEMA.title, Literal("The RDF Guide")))
    g.add((EX.book1, SCHEMA.author, EX.person1))
    g.add((EX.book1, SCHEMA.datePublished, Literal(2023)))
    g.add((EX.book1, SCHEMA.isbn, Literal("978-3-16-148410-0")))

    # Print the graph
    print("Sample RDF Graph:")
    print(g.serialize(format="turtle"))
    print()

    # Define a SPARQL query to find all people
    query = """
    SELECT ?person WHERE {
        ?person a foaf:Person .
    }
    """

    print("Running SPARQL query to find all people...")

    # Execute the query
    results = g.query(query)

    # Convert results to Person models
    people = []
    for result in results:
        person = Person.from_sparql_result(g, result)
        people.append(person)

    # Print the results
    print(f"Found {len(people)} people:")
    for person in people:
        print(f"  - {person.name} (URI: {person.uri})")
    print()

    # Find a specific book and its author with SPARQL
    book_query = """
    SELECT ?book ?title ?author WHERE {
        ?book a schema:Book ;
              schema:title ?title ;
              schema:author ?author .
    }
    """

    print("Running SPARQL query to find books with authors...")

    # Execute the query
    book_results = g.query(book_query)

    # Process the results
    for result in book_results:
        assert isinstance(result, ResultRow)

        book_uri = result.get("book")
        if book_uri is not None and isinstance(book_uri, URIRef):
            book = Book.parse_graph(g, book_uri)
            print(f"Book: {book.title}")
            print(f"Author: {book.author.name}")
            if book.year:
                print(f"Year: {book.year}")
            if book.isbn:
                print(f"ISBN: {book.isbn}")


if __name__ == "__main__":
    main()
