#!/usr/bin/env python3
"""
Advanced usage example for PydanticRDF (pydantic-rdf).

This example demonstrates:
1. Using validators and computed fields
2. Working with custom datatypes
3. Handling circular references
4. Inheritance in RDF models
"""

from datetime import datetime
from typing import Annotated, Self

from pydantic import Field, computed_field, field_validator, model_validator
from rdflib import Namespace
from rdflib.namespace import XSD

from pydantic_rdf import BaseRdfModel, WithDataType

# Define namespaces
EX = Namespace("http://example.org/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")


# Base class for all content types
class BaseContent(BaseRdfModel):
    """Base class for content items."""

    rdf_type = EX.Content
    _rdf_namespace = EX

    title: str
    created_at: datetime
    author: str

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty and has proper capitalization."""
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip().title()

    @computed_field
    def age_days(self) -> int:
        """Calculate the age of the content in days."""
        delta = datetime.now() - self.created_at
        return delta.days


# Article subclass
class Article(BaseContent):
    """A blog article."""

    rdf_type = EX.Article

    content: str
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content_length(self) -> Self:
        """Ensure content is substantial enough for an article."""
        if len(self.content) < 50:
            raise ValueError("Article content must be at least 50 characters")
        return self


# Video subclass
class Video(BaseContent):
    """A video content item."""

    rdf_type = EX.Video

    duration: Annotated[int, WithDataType(XSD.integer)]  # Duration in seconds
    url: str


# Example of circular references
class Node(BaseRdfModel):
    """A node in a linked list, demonstrating circular references."""

    rdf_type = EX.Node
    _rdf_namespace = EX

    name: str
    next: Self | None = None  # Self-reference


def main():
    # Create an article
    article = Article(
        uri=EX.article1,
        title="understanding rdf and pydantic",  # Will be capitalized
        content="This is an article about using PydanticRDF to bridge Pydantic and RDF graphs. " * 3,
        created_at=datetime(2025, 4, 20, 12, 0, 0),
        author="John Doe",
        tags=["RDF", "Pydantic", "Python", "Semantic Web"],
    )

    # Serialize to RDF
    graph = article.model_dump_rdf()

    print("Article RDF Graph:")
    print(graph.serialize(format="turtle"))
    print()

    # Load the article from RDF
    loaded_article = Article.parse_graph(graph, EX.article1)
    print(f"Loaded Article: {loaded_article.title}")
    print(f"Age in days: {loaded_article.age_days}")
    print(f"Tags: {', '.join(loaded_article.tags)}")
    print()

    # Create linked nodes demonstrating circular references
    node3 = Node(uri=EX.node3, name="Node 3", next=None)
    node2 = Node(uri=EX.node2, name="Node 2", next=node3)
    node1 = Node(uri=EX.node1, name="Node 1", next=node2)
    # Create a cycle
    node3.next = node1

    # Serialize to RDF
    graph2 = node1.model_dump_rdf()
    # Add the other nodes to the same graph
    graph2 += node2.model_dump_rdf()
    graph2 += node3.model_dump_rdf()

    print("Nodes RDF Graph:")
    print(graph2.serialize(format="turtle"))
    print()

    # Load the nodes from RDF
    loaded_node1 = Node.parse_graph(graph2, EX.node1)
    # Access the entire circular structure
    loaded_node2 = loaded_node1.next
    loaded_node3 = loaded_node2.next if loaded_node2 else None

    # Print the circular reference
    print("Circular Reference:")
    print(f"Node 1 -> {loaded_node1.name}")
    if loaded_node2:
        print(f"Node 2 -> {loaded_node2.name}")
    if loaded_node3:
        print(f"Node 3 -> {loaded_node3.name}")
        if loaded_node3.next:
            print(f"Node 3 -> Node 1? {loaded_node3.next.uri == EX.node1}")


if __name__ == "__main__":
    main()
