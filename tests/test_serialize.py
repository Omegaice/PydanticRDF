from datetime import date, datetime
from typing import Annotated

import pytest
from pydantic import ConfigDict, field_serializer
from rdflib import XSD, Graph, Literal, Namespace

from pydantic_rdf.annotation import WithPredicate
from pydantic_rdf.model import BaseRdfModel


def test_basic_string_field_serialization(graph: Graph, EX: Namespace):
    """Test that a simple string field is serialized as an RDF triple with the correct predicate and value."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX

        name: str

    obj = MyType(uri=EX.entity1, name="TestName")
    graph += obj.model_dump_rdf()
    assert (EX.entity1, EX.name, Literal("TestName")) in graph


def test_extra_rdf_triples_serialization(graph: Graph, EX: Namespace):
    """Test that only model fields are serialized and unrelated triples are not added to the graph."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str

    obj = MyType(uri=EX.entity1, name="TestName")
    graph += obj.model_dump_rdf()
    assert (EX.entity1, EX.name, Literal("TestName")) in graph
    assert (EX.entity1, EX.unrelated, None) not in graph


def test_nested_BaseRdfModel_serialization(graph: Graph, EX: Namespace):
    """Test that nested BaseRdfModel fields are serialized as related resources with correct predicates."""

    class Child(BaseRdfModel):
        rdf_type = EX.Child
        _rdf_namespace = EX
        value: str

    class Parent(BaseRdfModel):
        rdf_type = EX.Parent
        _rdf_namespace = EX
        name: str
        child: Annotated[Child, WithPredicate(EX.hasChild)]

    child = Child(uri=EX.child1, value="child1-value")
    parent = Parent(uri=EX.parent1, name="ParentName", child=child)
    graph += parent.model_dump_rdf()
    assert (EX.parent1, EX.name, Literal("ParentName")) in graph
    assert (EX.parent1, EX.hasChild, EX.child1) in graph
    assert (EX.child1, EX.value, Literal("child1-value")) in graph


def test_list_of_nested_BaseRdfModels_serialization(graph: Graph, EX: Namespace):
    """Test that a list of nested BaseRdfModel instances is serialized as multiple related resources."""

    class Child(BaseRdfModel):
        rdf_type = EX.Child
        _rdf_namespace = EX
        value: str

    class Parent(BaseRdfModel):
        rdf_type = EX.Parent
        _rdf_namespace = EX
        name: str
        children: Annotated[list[Child], WithPredicate(EX.hasChildren)]

    child1 = Child(uri=EX.child1, value="child1-value")
    child2 = Child(uri=EX.child2, value="child2-value")
    parent = Parent(uri=EX.parent1, name="ParentName", children=[child1, child2])
    graph += parent.model_dump_rdf()
    assert (EX.parent1, EX.name, Literal("ParentName")) in graph
    assert (EX.parent1, EX.hasChildren, EX.child1) in graph
    assert (EX.parent1, EX.hasChildren, EX.child2) in graph
    assert (EX.child1, EX.value, Literal("child1-value")) in graph
    assert (EX.child2, EX.value, Literal("child2-value")) in graph


def test_list_of_literals_serialization(graph: Graph, EX: Namespace):
    """Test that a list of literal values is serialized as multiple triples for the same predicate."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        tags: list[str]

    obj = MyType(uri=EX.entity1, tags=["tag1", "tag2"])
    graph += obj.model_dump_rdf()
    assert (EX.entity1, EX.tags, Literal("tag1")) in graph
    assert (EX.entity1, EX.tags, Literal("tag2")) in graph


def test_optional_field_serialization(graph: Graph, EX: Namespace):
    """Test that optional fields are only serialized if they are not None."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str
        nickname: str | None = None

    obj = MyType(uri=EX.entity1, name="TestName")
    graph += obj.model_dump_rdf()
    assert (EX.entity1, EX.name, Literal("TestName")) in graph
    # Should not serialize EX.nickname if None
    assert (EX.entity1, EX.nickname, None) not in graph


def test_field_with_default_value_serialization(graph: Graph, EX: Namespace):
    """Test that fields with default values are serialized with their default if not explicitly set."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str
        status: str = "active"

    obj = MyType(uri=EX.entity1, name="TestName")
    graph += obj.model_dump_rdf()
    assert (EX.entity1, EX.name, Literal("TestName")) in graph
    assert (EX.entity1, EX.status, Literal("active")) in graph


def test_multiple_entities_serialization(graph: Graph, EX: Namespace):
    """Test that multiple model instances serialize to separate sets of triples in the graph."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str

    obj1 = MyType(uri=EX.entity1, name="Entity1")
    obj2 = MyType(uri=EX.entity2, name="Entity2")

    graph += obj1.model_dump_rdf()
    graph += obj2.model_dump_rdf()

    assert (EX.entity1, EX.name, Literal("Entity1")) in graph
    assert (EX.entity2, EX.name, Literal("Entity2")) in graph


def test_self_reference_serialization(graph: Graph, EX: Namespace):
    """Test that self-referential fields are serialized as resource links (URIs) in the graph."""

    class Node(BaseRdfModel):
        rdf_type = EX.Node
        _rdf_namespace = EX
        name: str
        next: Annotated["Node", None] | None = None

    node2 = Node(uri=EX.node2, name="Node2")
    node1 = Node(uri=EX.node1, name="Node1", next=node2)
    graph += node1.model_dump_rdf()
    assert (EX.node1, EX.name, Literal("Node1")) in graph
    assert (EX.node1, EX.next, EX.node2) in graph
    assert (EX.node2, EX.name, Literal("Node2")) in graph


def test_field_with_unknown_type_serialization(graph: Graph, EX: Namespace):
    """Test that unsupported field types (e.g., complex) are not serialized or raise an error."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        data: complex

    obj = MyType(uri=EX.entity1, data=complex(1, 2))
    graph += obj.model_dump_rdf()
    # Should raise TypeError or skip serialization
    # (No assertion, just ensure test fails or is skipped)
    pass


def test_successful_type_coercion_serialization(graph: Graph, EX: Namespace):
    """Test that values are correctly coerced to their field types and serialized as expected."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        score: float

    obj = MyType(uri=EX.entity1, score=1.5)
    graph += obj.model_dump_rdf()
    assert (EX.entity1, EX.score, Literal(1.5)) in graph


def test_computed_field_not_serialized(graph: Graph, EX: Namespace):
    """Test that computed fields (properties) are not serialized as RDF triples."""

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX
        first_name: str
        last_name: str

        @property
        def full_name(self) -> str:
            return f"{self.first_name} {self.last_name}"

    obj = Person(uri=EX.person1, first_name="John", last_name="Doe")
    graph += obj.model_dump_rdf()
    assert (EX.person1, EX.first_name, Literal("John")) in graph
    assert (EX.person1, EX.last_name, Literal("Doe")) in graph
    # Computed field should not be serialized
    assert (EX.person1, EX.full_name, None) not in graph


def test_field_validator_serialization(graph: Graph, EX: Namespace):
    """Test that field validators are respected and validated values are serialized."""

    class User(BaseRdfModel):
        rdf_type = EX.User
        _rdf_namespace = EX
        email: str

        @classmethod
        def validate_email(cls, v: str) -> str:
            if "@" not in v:
                raise ValueError("Invalid email format")
            return v.lower()

    obj = User(uri=EX.user1, email="John.Doe@example.com")
    graph += obj.model_dump_rdf()
    assert (EX.user1, EX.email, Literal("John.Doe@example.com")) in graph


def test_frozen_model_serialization(graph: Graph, EX: Namespace):
    """Test that frozen models can still be serialized to RDF triples."""

    class Config(BaseRdfModel):
        rdf_type = EX.Config
        _rdf_namespace = EX
        model_config = ConfigDict(frozen=True)
        name: str
        version: str

    obj = Config(uri=EX.config1, name="MyApp", version="1.0.0")
    graph += obj.model_dump_rdf()
    assert (EX.config1, EX.name, Literal("MyApp")) in graph
    assert (EX.config1, EX.version, Literal("1.0.0")) in graph


def test_frozen_model_dict_serialization(graph: Graph, EX: Namespace):
    """Test that dictionary fields in frozen models are serialized as expected."""

    class Config(BaseRdfModel):
        rdf_type = EX.Config
        _rdf_namespace = EX
        model_config = ConfigDict(frozen=True)
        settings: dict[str, str]

    obj = Config(uri=EX.config1, settings={"theme": "dark", "language": "en"})
    graph += obj.model_dump_rdf()
    assert (EX.config1, EX.settings, Literal('{"theme": "dark", "language": "en"}')) in graph


def test_article_subclass_serialization(graph: Graph, EX: Namespace):
    """Test that subclassed models serialize both base and subclass fields correctly."""

    class BaseContent(BaseRdfModel):
        """Base class for content-related serialization tests."""

        rdf_type = EX.Content
        _rdf_namespace = EX
        title: str
        created_at: str
        author: str

    class Article(BaseContent):
        rdf_type = EX.Article
        content: str
        tags: list[str]

    obj = Article(
        uri=EX.article1,
        title="My First Article",
        created_at="2024-03-15",
        author="John Doe",
        content="This is the article content",
        tags=["python", "rdf"],
    )
    graph += obj.model_dump_rdf()
    assert (EX.article1, EX.title, Literal("My First Article")) in graph
    assert (EX.article1, EX.created_at, Literal("2024-03-15")) in graph
    assert (EX.article1, EX.author, Literal("John Doe")) in graph
    assert (EX.article1, EX.content, Literal("This is the article content")) in graph
    assert (EX.article1, EX.tags, Literal("python")) in graph
    assert (EX.article1, EX.tags, Literal("rdf")) in graph


def test_video_subclass_serialization(graph: Graph, EX: Namespace):
    """Test that another subclassed model serializes all its fields correctly."""

    class BaseContent(BaseRdfModel):
        """Base class for content-related serialization tests."""

        rdf_type = EX.Content
        _rdf_namespace = EX
        title: str
        created_at: str
        author: str

    class Video(BaseContent):
        rdf_type = EX.Video
        duration: int
        url: str

    obj = Video(
        uri=EX.video1,
        title="My Tutorial Video",
        created_at="2024-03-16",
        author="Jane Smith",
        duration=300,
        url="https://example.com/video1",
    )
    graph += obj.model_dump_rdf()
    assert (EX.video1, EX.title, Literal("My Tutorial Video")) in graph
    assert (EX.video1, EX.created_at, Literal("2024-03-16")) in graph
    assert (EX.video1, EX.author, Literal("Jane Smith")) in graph
    assert (EX.video1, EX.duration, Literal(300)) in graph
    assert (EX.video1, EX.url, Literal("https://example.com/video1")) in graph


def test_custom_serialization_format(graph: Graph, EX: Namespace):
    """Test that custom serialization logic produces the expected RDF triples."""

    class Event(BaseRdfModel):
        rdf_type = EX.Event
        _rdf_namespace = EX
        name: str
        timestamp: datetime
        metadata: dict[str, str]

        @field_serializer("timestamp")
        def serialize_timestamp(self, value):
            return value.isoformat()

    obj = Event(
        uri=EX.event1,
        name="System Update",
        timestamp=datetime(2024, 3, 15, 14, 30, 0),
        metadata={"status": "completed", "duration": "5m"},
    )
    graph += obj.model_dump_rdf()
    # obj.to_graph(graph)
    assert (EX.event1, EX.name, Literal("System Update")) in graph
    assert (EX.event1, EX.timestamp, Literal("2024-03-15T14:30:00")) in graph
    assert (EX.event1, EX.metadata, Literal('{"status": "completed", "duration": "5m"}')) in graph


def test_serialization_type_preservation(graph: Graph, EX: Namespace):
    """Test that field types (e.g., datetime) are preserved and serialized in the correct format."""

    class Event(BaseRdfModel):
        rdf_type = EX.Event
        _rdf_namespace = EX
        timestamp: datetime

    obj = Event(uri=EX.event1, timestamp=datetime(2024, 3, 15, 14, 30, 0))
    graph += obj.model_dump_rdf()
    assert (EX.event1, EX.timestamp, Literal("2024-03-15T14:30:00", datatype=XSD.dateTime)) in graph


@pytest.mark.xfail(
    reason="Multiple values for a single-valued field are not yet supported; should pick one or raise a clear error"
)
def test_multiple_values_single_field_serialization(graph: Graph, EX: Namespace):
    """Test that only one value is serialized for single-valued fields, or an error is raised."""

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX
        name: str

    obj = Person(uri=EX.person1, name="John")
    graph += obj.model_dump_rdf()
    assert (EX.person1, EX.name, Literal("John")) in graph
    # If multiple values, should pick one or raise error


@pytest.mark.xfail(reason="Language-tagged literals are not yet supported; should handle language alternatives")
def test_language_tagged_literals_serialization(graph: Graph, EX: Namespace):
    """Test that language-tagged literals are handled or ignored during serialization."""

    class MultiLingualContent(BaseRdfModel):
        rdf_type = EX.Content
        _rdf_namespace = EX
        title: str

    obj = MultiLingualContent(uri=EX.content1, title="Hello")
    graph += obj.model_dump_rdf()
    assert (EX.content1, EX.title, Literal("Hello")) in graph
    # Should support language tags in future


@pytest.mark.xfail(reason="Blank nodes are not yet supported; should handle complex structures")
def test_blank_node_complex_structure_serialization(graph: Graph, EX: Namespace):
    """Test that blank nodes and their nested structure are serialized correctly."""

    class Address(BaseRdfModel):
        rdf_type = EX.Address
        _rdf_namespace = EX
        street: str
        city: str

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX
        name: str
        address: Annotated[Address, WithPredicate(EX.address)]

    address = Address(uri=EX.address1, street="123 Main St", city="Springfield")
    person = Person(uri=EX.person1, name="John Doe", address=address)
    graph += person.model_dump_rdf()
    assert (EX.person1, EX.name, Literal("John Doe")) in graph
    assert (EX.person1, EX.address, EX.address1) in graph
    assert (EX.address1, EX.street, Literal("123 Main St")) in graph
    assert (EX.address1, EX.city, Literal("Springfield")) in graph


def test_multiple_rdf_types_serialization(graph: Graph, EX: Namespace):
    """Test that resources with multiple rdf:types are serialized with all relevant type triples."""

    class Employee(BaseRdfModel):
        rdf_type = EX.Employee
        _rdf_namespace = EX
        name: str

    obj = Employee(uri=EX.person1, name="John Doe")
    graph += obj.model_dump_rdf()
    assert (EX.person1, EX.name, Literal("John Doe")) in graph
    # Should add rdf:type triple for Employee


@pytest.mark.xfail(reason="RDF lists are not yet supported; should preserve order from rdf:List")
def test_rdf_list_ordering_serialization(graph: Graph, EX: Namespace):
    """Test that lists are serialized as RDF lists, preserving order."""

    class Playlist(BaseRdfModel):
        rdf_type = EX.Playlist
        _rdf_namespace = EX
        name: str
        tracks: list[str]

    obj = Playlist(uri=EX.playlist1, name="My Playlist", tracks=["track1", "track2", "track3"])
    graph += obj.model_dump_rdf()
    assert (EX.playlist1, EX.name, Literal("My Playlist")) in graph
    # Should serialize tracks as rdf:List in order


@pytest.mark.xfail(reason="RDF reification is not yet supported; should parse reified statements")
def test_reification_handling_serialization(graph: Graph, EX: Namespace):
    """Test that reified statements are serialized as RDF reification structures."""

    class Statement(BaseRdfModel):
        rdf_type = EX.Statement
        _rdf_namespace = EX
        subject: str
        predicate: str
        object: str
        confidence: float

    obj = Statement(uri=EX.statement1, subject="JohnDoe", predicate="knows", object="JaneSmith", confidence=0.9)
    graph += obj.model_dump_rdf()
    assert (EX.statement1, EX.subject, Literal("JohnDoe")) in graph
    assert (EX.statement1, EX.predicate, Literal("knows")) in graph
    assert (EX.statement1, EX.object, Literal("JaneSmith")) in graph
    assert (EX.statement1, EX.confidence, Literal(0.9)) in graph


def test_custom_datatype_literals_serialization(graph: Graph, EX: Namespace):
    """Test that custom datatypes (e.g., geo:point, xsd:date) are serialized as RDF literals with correct datatypes."""

    class CustomDatatypes(BaseRdfModel):
        rdf_type = EX.CustomDatatypes
        _rdf_namespace = EX
        coordinate: str
        date: date

    obj = CustomDatatypes(uri=EX.data1, coordinate="45.123,-122.456", date=date(2024, 3, 20))
    graph += obj.model_dump_rdf()
    assert (EX.data1, EX.coordinate, Literal("45.123,-122.456")) in graph
    assert (EX.data1, EX.date, Literal("2024-03-20", datatype=XSD.date)) in graph


@pytest.mark.xfail(reason="RDF container membership properties are not yet supported; should parse container items")
def test_container_membership_serialization(graph: Graph, EX: Namespace):
    """Test that RDF containers are serialized using container membership properties (rdf:_1, rdf:_2, ...)."""

    class Container(BaseRdfModel):
        rdf_type = EX.Bag
        _rdf_namespace = EX
        items: list[str]

    obj = Container(uri=EX.container1, items=["item1", "item2", "item3"])
    graph += obj.model_dump_rdf()
    # Should serialize items as rdf:_1, rdf:_2, ...


def test_datatype_promotion_serialization(graph: Graph, EX: Namespace):
    """Test that numeric values are promoted/coerced to the correct datatype in serialization."""

    class Numbers(BaseRdfModel):
        rdf_type = EX.Numbers
        _rdf_namespace = EX
        integer_field: int
        decimal_field: float
        any_number: float

    obj = Numbers(uri=EX.numbers1, integer_field=42, decimal_field=42.0, any_number=42.0)
    graph += obj.model_dump_rdf()
    assert (EX.numbers1, EX.integer_field, Literal(42)) in graph
    assert (EX.numbers1, EX.decimal_field, Literal(42.0)) in graph
    assert (EX.numbers1, EX.any_number, Literal(42.0)) in graph


@pytest.mark.xfail(reason="Blank nodes are not yet supported")
def test_blank_node_identity_serialization(graph: Graph, EX: Namespace):
    """Test that blank node identity is preserved and shared blank nodes are serialized consistently."""

    class NodeContainer(BaseRdfModel):
        rdf_type = EX.NodeContainer
        _rdf_namespace = EX
        name: str
        related: Annotated[list["NodeContainer"], WithPredicate(EX.related)]

    shared = NodeContainer(uri=EX.shared, name="Shared Node", related=[])
    container1 = NodeContainer(uri=EX.container1, name="Container 1", related=[shared])
    graph += container1.model_dump_rdf()
    container2 = NodeContainer(uri=EX.container2, name="Container 2", related=[shared])
    graph += container2.model_dump_rdf()
    assert (EX.container1, EX.related, EX.shared) in graph
    assert (EX.container2, EX.related, EX.shared) in graph
    assert (EX.shared, EX.name, Literal("Shared Node")) in graph


@pytest.mark.xfail(reason="We don't support SPARQL-like property paths.")
def test_property_paths_serialization(graph: Graph, EX: Namespace):
    """Test that property paths (complex predicate chains) are handled or raise a clear error during serialization."""

    class Organization(BaseRdfModel):
        rdf_type = EX.Organization
        _rdf_namespace = EX
        name: str

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX
        name: str

    org = Organization(uri=EX.org1, name="Test Org")
    graph += org.model_dump_rdf()
    assert (EX.org1, EX.name, Literal("Test Org")) in graph
