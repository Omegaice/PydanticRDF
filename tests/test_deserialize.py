from typing import Annotated, Self

import pytest
from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace
from rdflib.extras.describer import Describer

from pydantic_rdf.annotation import WithPredicate
from pydantic_rdf.model import BaseRdfModel


def test_basic_string_field(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX

        name: str

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.name, Literal("TestName"))

    # Deserialize the entity from the graph
    obj = MyType.parse_graph(graph, EX.entity1)
    assert obj.name == "TestName"


def test_missing_required_field(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    # No EX.name triple

    # Deserialize the entity from the graph and assert results
    with pytest.raises(ValidationError):
        MyType.parse_graph(graph, EX.entity1)


def test_extra_rdf_triples(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.name, Literal("TestName"))
    d.value(EX.unrelated, Literal("ShouldBeIgnored"))

    # Deserialize the entity from the graph
    obj = MyType.parse_graph(graph, EX.entity1)
    assert obj.name == "TestName"


def test_incorrect_rdf_type(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.OtherType)  # Wrong type!
    d.value(EX.name, Literal("TestName"))

    # Deserialize the entity from the graph and assert results
    with pytest.raises(ValueError):
        MyType.parse_graph(graph, EX.entity1)


def test_nested_BaseRdfModel_extraction(graph: Graph, EX: Namespace):
    # Define class to test
    class Child(BaseRdfModel):
        rdf_type = EX.Child
        _rdf_namespace = EX
        value: str

    class Parent(BaseRdfModel):
        rdf_type = EX.Parent
        _rdf_namespace = EX
        name: str
        child: Annotated[Child, WithPredicate(EX.hasChild)]

    # Create a new entity
    d = Describer(graph=graph, about=EX.parent1)
    d.rdftype(EX.Parent)
    d.value(EX.name, Literal("ParentName"))

    # Add a child relationship
    with d.rel(EX.hasChild, EX.child1):
        d.rdftype(EX.Child)
        d.value(EX.value, Literal("child1-value"))

    # Deserialize the entity from the graph
    parent = Parent.parse_graph(graph, EX.parent1)

    # Assert results
    assert parent.name == "ParentName"
    assert isinstance(parent.child, Child)
    assert parent.child.value == "child1-value"


def test_list_of_nested_BaseRdfModels(graph: Graph, EX: Namespace):
    # Define class to test
    class Child(BaseRdfModel):
        rdf_type = EX.Child
        _rdf_namespace = EX

        value: str

    class Parent(BaseRdfModel):
        rdf_type = EX.Parent
        _rdf_namespace = EX

        name: str
        children: Annotated[list[Child], WithPredicate(EX.hasChildren)] = Field(default_factory=list)

    # Create a new entity
    d = Describer(graph=graph, about=EX.parent1)
    d.rdftype(EX.Parent)
    d.value(EX.name, Literal("ParentName"))

    with d.rel(EX.hasChildren, EX.child1):
        d.rdftype(EX.Child)
        d.value(EX.value, Literal("child1-value"))

    with d.rel(EX.hasChildren, EX.child2):
        d.rdftype(EX.Child)
        d.value(EX.value, Literal("child2-value"))

    # Deserialize the entity from the graph
    parent = Parent.parse_graph(graph, EX.parent1)

    # Assert results
    assert parent.name == "ParentName"
    assert isinstance(parent.children, list)
    assert {c.value for c in parent.children} == {"child1-value", "child2-value"}


def test_list_of_literals(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        tags: list[str]

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.tags, Literal("tag1"))
    d.value(EX.tags, Literal("tag2"))

    # Deserialize the entity from the graph
    obj = MyType.parse_graph(graph, EX.entity1)

    # Assert results
    assert isinstance(obj.tags, list)
    assert set(obj.tags) == {"tag1", "tag2"}


def test_default_predicate_no_annotation(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        description: str

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.description, Literal("A description"))

    # Deserialize the entity from the graph
    obj = MyType.parse_graph(graph, EX.entity1)
    # Assert results
    assert obj.description == "A description"


def test_optional_field(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str
        nickname: str | None = None

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.name, Literal("TestName"))
    # No EX.nickname triple

    # Deserialize the entity from the graph
    obj = MyType.parse_graph(graph, EX.entity1)
    # Assert results
    assert obj.name == "TestName"
    assert obj.nickname is None


def test_field_with_default_value(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str
        status: str = "active"

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.name, Literal("TestName"))
    # No EX.status triple

    # Deserialize the entity from the graph
    obj = MyType.parse_graph(graph, EX.entity1)
    # Assert results
    assert obj.name == "TestName"
    assert obj.status == "active"


def test_multiple_entities_extraction(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        name: str

    # Create new entities
    d1 = Describer(graph=graph, about=EX.entity1)
    d1.rdftype(EX.MyType)
    d1.value(EX.name, Literal("Entity1"))

    d2 = Describer(graph=graph, about=EX.entity2)
    d2.rdftype(EX.MyType)
    d2.value(EX.name, Literal("Entity2"))

    # Deserialize all entities from the graph
    objs = MyType.all_entities(graph)
    # Assert results
    names = {obj.name for obj in objs}
    assert names == {"Entity1", "Entity2"}


def test_non_uriref_subject(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX

        name: str

    # Create a new entity with a BNode (not a URIRef)
    bnode = BNode()
    graph.add((bnode, EX.name, Literal("TestName")))
    graph.add((bnode, EX.rdftype, EX.MyType))

    # Should not return any entities for BNode
    objs = MyType.all_entities(graph)

    # Assert results
    assert objs == []


def test_self_reference(graph: Graph, EX: Namespace):
    # Define class to test
    class Node(BaseRdfModel):
        rdf_type = EX.Node
        _rdf_namespace = EX
        name: str
        next: Self | None = None

    # Create two nodes referencing each other
    d = Describer(graph=graph, about=EX.node1)
    d.rdftype(EX.Node)
    d.value(EX.name, Literal("Node1"))
    with d.rel(EX.next, EX.node2):
        d.rdftype(EX.Node)
        d.value(EX.name, Literal("Node2"))

    # Deserialize the entity from the graph
    node1 = Node.parse_graph(graph, EX.node1)

    # Assert results
    assert node1.name == "Node1"
    assert node1.next is not None  # Add type check before accessing attribute
    assert node1.next.name == "Node2"


def test_circular_reference(graph: Graph, EX: Namespace):
    # Define class to test
    class Node(BaseRdfModel):
        rdf_type = EX.Node
        _rdf_namespace = EX
        name: str
        next: Self | None = None

    # Create two nodes referencing each other
    d = Describer(graph=graph, about=EX.node1)
    d.rdftype(EX.Node)
    d.value(EX.name, Literal("Node1"))
    with d.rel(EX.next, EX.node2):
        d.rdftype(EX.Node)
        d.value(EX.name, Literal("Node2"))
        d.rel(EX.next, EX.node1)

    # Deserialize the entity from the graph
    with pytest.raises(RecursionError):
        Node.parse_graph(graph, EX.node1)


def test_field_with_unknown_type(graph: Graph, EX: Namespace):
    # Define class to test
    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        data: complex  # Not supported

    # Create a new entity
    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.data, Literal("not-a-complex"))

    # Deserialize the entity from the graph and assert results
    with pytest.raises(TypeError):
        MyType.parse_graph(graph, EX.entity1)


def test_failed_type_coercion(graph: Graph, EX: Namespace):
    """Test that invalid type coercion raises appropriate validation error."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        age: int

    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.age, Literal("not-an-integer"))

    with pytest.raises(ValidationError) as exc_info:
        MyType.parse_graph(graph, EX.entity1)

    assert "age" in str(exc_info.value)


def test_successful_type_coercion(graph: Graph, EX: Namespace):
    """Test that valid string-to-number coercion works."""

    class MyType(BaseRdfModel):
        rdf_type = EX.MyType
        _rdf_namespace = EX
        score: float

    d = Describer(graph=graph, about=EX.entity1)
    d.rdftype(EX.MyType)
    d.value(EX.score, Literal("1.5"))

    obj = MyType.parse_graph(graph, EX.entity1)
    assert obj.score == 1.5
    assert isinstance(obj.score, float)


def test_computed_field_string_concatenation(graph: Graph, EX: Namespace):
    """Test computed field that concatenates string values."""

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX

        first_name: str
        last_name: str

        @computed_field
        def full_name(self) -> str:
            return f"{self.first_name} {self.last_name}"

    d = Describer(graph=graph, about=EX.person1)
    d.rdftype(EX.Person)
    d.value(EX.first_name, Literal("John"))
    d.value(EX.last_name, Literal("Doe"))

    person = Person.parse_graph(graph, EX.person1)
    assert person.full_name == "John Doe"


def test_computed_field_boolean_condition(graph: Graph, EX: Namespace):
    """Test computed field that evaluates a boolean condition."""

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX

        age: int

        @computed_field
        def is_adult(self) -> bool:
            return self.age >= 18

    # Test with adult
    d1 = Describer(graph=graph, about=EX.person1)
    d1.rdftype(EX.Person)
    d1.value(EX.age, Literal(25))

    adult = Person.parse_graph(graph, EX.person1)
    assert adult.is_adult is True

    # Test with minor
    d2 = Describer(graph=graph, about=EX.person2)
    d2.rdftype(EX.Person)
    d2.value(EX.age, Literal(15))

    minor = Person.parse_graph(graph, EX.person2)
    assert minor.is_adult is False


def test_email_field_validation(graph: Graph, EX: Namespace):
    """Test email field validation with custom validator."""

    class User(BaseRdfModel):
        rdf_type = EX.User
        _rdf_namespace = EX

        email: str

        @field_validator("email")
        @classmethod
        def validate_email(cls, v: str) -> str:
            if "@" not in v:
                raise ValueError("Invalid email format")
            return v.lower()

    # Test valid email
    d1 = Describer(graph=graph, about=EX.user1)
    d1.rdftype(EX.User)
    d1.value(EX.email, Literal("John.Doe@example.com"))

    user = User.parse_graph(graph, EX.user1)
    assert user.email == "john.doe@example.com"  # Check email was lowercased

    # Test invalid email
    d2 = Describer(graph=graph, about=EX.user2)
    d2.rdftype(EX.User)
    d2.value(EX.email, Literal("invalid-email"))

    with pytest.raises(ValidationError) as exc_info:
        User.parse_graph(graph, EX.user2)
    assert "Invalid email format" in str(exc_info.value)


def test_numeric_field_validation(graph: Graph, EX: Namespace):
    """Test numeric field validation with custom validator."""

    class Payment(BaseRdfModel):
        rdf_type = EX.Payment
        _rdf_namespace = EX

        amount: float

        @field_validator("amount")
        @classmethod
        def validate_amount(cls, v: float) -> float:
            if v < 0:
                raise ValueError("Amount cannot be negative")
            return v

    # Test valid amount
    d1 = Describer(graph=graph, about=EX.payment1)
    d1.rdftype(EX.Payment)
    d1.value(EX.amount, Literal(100.0))

    payment = Payment.parse_graph(graph, EX.payment1)
    assert payment.amount == 100.0

    # Test negative amount
    d2 = Describer(graph=graph, about=EX.payment2)
    d2.rdftype(EX.Payment)
    d2.value(EX.amount, Literal(-50.0))

    with pytest.raises(ValidationError) as exc_info:
        Payment.parse_graph(graph, EX.payment2)
    assert "Amount cannot be negative" in str(exc_info.value)


def test_model_level_validation(graph: Graph, EX: Namespace):
    """Test model-level validation with custom validator."""

    class Employee(BaseRdfModel):
        rdf_type = EX.Employee
        _rdf_namespace = EX

        department: str
        salary: float

        @model_validator(mode="after")
        def validate_department_salary(self) -> Self:
            if self.department == "IT" and self.salary < 50000:
                raise ValueError("IT department salary must be at least 50000")
            return self

    # Test valid case
    d1 = Describer(graph=graph, about=EX.emp1)
    d1.rdftype(EX.Employee)
    d1.value(EX.department, Literal("IT"))
    d1.value(EX.salary, Literal(60000))

    emp = Employee.parse_graph(graph, EX.emp1)
    assert emp.department == "IT"
    assert emp.salary == 60000

    # Test invalid case
    d2 = Describer(graph=graph, about=EX.emp2)
    d2.rdftype(EX.Employee)
    d2.value(EX.department, Literal("IT"))
    d2.value(EX.salary, Literal(40000))

    with pytest.raises(ValidationError) as exc_info:
        Employee.parse_graph(graph, EX.emp2)
    assert "IT department salary must be at least 50000" in str(exc_info.value)


def test_frozen_model_attribute_modification(graph: Graph, EX: Namespace):
    """Test that frozen models prevent direct attribute modification."""

    class Config(BaseRdfModel):
        rdf_type = EX.Config
        _rdf_namespace = EX

        model_config = ConfigDict(frozen=True)

        name: str
        version: str

    d = Describer(graph=graph, about=EX.config1)
    d.rdftype(EX.Config)
    d.value(EX.name, Literal("MyApp"))
    d.value(EX.version, Literal("1.0.0"))

    config = Config.parse_graph(graph, EX.config1)

    with pytest.raises(ValidationError) as exc_info:
        config.name = "NewName"
    assert "frozen" in str(exc_info.value).lower()

    assert config.name == "MyApp"
    assert config.version == "1.0.0"


def test_frozen_model_dict_modification(graph: Graph, EX: Namespace):
    """Test that frozen models prevent dictionary field modification."""

    class Config(BaseRdfModel):
        rdf_type = EX.Config
        _rdf_namespace = EX

        model_config = ConfigDict(frozen=True)

        settings: dict[str, str] = Field(default_factory=dict)

    d = Describer(graph=graph, about=EX.config1)
    d.rdftype(EX.Config)
    d.value(EX.settings, Literal('{"theme": "dark", "language": "en"}'))

    config = Config.parse_graph(graph, EX.config1)

    # In Pydantic v2, frozen models raise ValidationError for any modification
    assert config.settings == {"theme": "dark", "language": "en"}
    with pytest.raises(ValidationError) as exc_info:
        config.settings = {"theme": "light"}  # Try to replace entire dict
    assert "frozen" in str(exc_info.value).lower()

    # Original data should be unchanged
    assert config.settings == {"theme": "dark", "language": "en"}


def test_article_subclass_parsing(graph: Graph, EX: Namespace):
    """Test parsing of Article subclass with its specific fields."""

    class BaseContent(BaseRdfModel):
        """Base class for content-related test models."""

        rdf_type = EX.Content
        _rdf_namespace = EX

        title: str
        created_at: str
        author: str

    class Article(BaseContent):
        rdf_type = EX.Article

        content: str
        tags: list[str] = Field(default_factory=list)

    d = Describer(graph=graph, about=EX.article1)
    d.rdftype(EX.Article)
    d.value(EX.title, Literal("My First Article"))
    d.value(EX.created_at, Literal("2024-03-15"))
    d.value(EX.author, Literal("John Doe"))
    d.value(EX.content, Literal("This is the article content"))
    d.value(EX.tags, Literal("python"))
    d.value(EX.tags, Literal("rdf"))

    article = Article.parse_graph(graph, EX.article1)
    assert article.title == "My First Article"
    assert article.created_at == "2024-03-15"
    assert article.author == "John Doe"
    assert article.content == "This is the article content"
    assert set(article.tags) == {"python", "rdf"}


def test_video_subclass_parsing(graph: Graph, EX: Namespace):
    """Test parsing of Video subclass with its specific fields."""

    class BaseContent(BaseRdfModel):
        """Base class for content-related test models."""

        rdf_type = EX.Content
        _rdf_namespace = EX

        title: str
        created_at: str
        author: str

    class Video(BaseContent):
        rdf_type = EX.Video

        duration: int
        url: str

    d = Describer(graph=graph, about=EX.video1)
    d.rdftype(EX.Video)
    d.value(EX.title, Literal("My Tutorial Video"))
    d.value(EX.created_at, Literal("2024-03-16"))
    d.value(EX.author, Literal("Jane Smith"))
    d.value(EX.duration, Literal(300))
    d.value(EX.url, Literal("https://example.com/video1"))

    video = Video.parse_graph(graph, EX.video1)
    assert video.title == "My Tutorial Video"
    assert video.created_at == "2024-03-16"
    assert video.author == "Jane Smith"
    assert video.duration == 300
    assert video.url == "https://example.com/video1"


def test_base_class_type_validation(graph: Graph, EX: Namespace):
    """Test that base class cannot parse entities of specific types."""

    class BaseContent(BaseRdfModel):
        """Base class for content-related test models."""

        rdf_type = EX.Content
        _rdf_namespace = EX

        title: str
        created_at: str
        author: str

    d = Describer(graph=graph, about=EX.content1)
    d.rdftype(EX.Article)
    d.value(EX.title, Literal("Test"))
    d.value(EX.created_at, Literal("2024-03-17"))
    d.value(EX.author, Literal("Test Author"))

    with pytest.raises(ValueError) as exc_info:
        BaseContent.parse_graph(graph, EX.content1)
    assert "type" in str(exc_info.value).lower()


def test_custom_serialization_format(graph: Graph, EX: Namespace):
    """Test that custom serialization produces the expected format."""
    from datetime import datetime

    class Event(BaseRdfModel):
        rdf_type = EX.Event
        _rdf_namespace = EX

        name: str
        timestamp: datetime
        metadata: dict[str, str] = Field(default_factory=dict)

        @model_serializer
        def serialize_model(self) -> dict:
            return {
                "name": self.name,
                "timestamp": self.timestamp.isoformat(),
                "metadata": self.metadata,
                "uri": str(self.uri),
            }

    d = Describer(graph=graph, about=EX.event1)
    d.rdftype(EX.Event)
    d.value(EX.name, Literal("System Update"))
    d.value(EX.timestamp, Literal("2024-03-15T14:30:00"))
    d.value(EX.metadata, Literal('{"status": "completed", "duration": "5m"}'))

    event = Event.parse_graph(graph, EX.event1)
    serialized = event.model_dump()

    assert serialized == {
        "name": "System Update",
        "timestamp": "2024-03-15T14:30:00",
        "metadata": {"status": "completed", "duration": "5m"},
        "uri": str(EX.event1),
    }


def test_serialization_type_preservation(graph: Graph, EX: Namespace):
    """Test that original types are preserved after serialization/deserialization."""
    from datetime import datetime

    class Event(BaseRdfModel):
        rdf_type = EX.Event
        _rdf_namespace = EX

        timestamp: datetime

    d = Describer(graph=graph, about=EX.event1)
    d.rdftype(EX.Event)
    d.value(EX.timestamp, Literal("2024-03-15T14:30:00"))

    event = Event.parse_graph(graph, EX.event1)

    assert isinstance(event.timestamp, datetime)
    assert event.timestamp.year == 2024
    assert event.timestamp.month == 3
    assert event.timestamp.day == 15
    assert event.timestamp.hour == 14
    assert event.timestamp.minute == 30


@pytest.mark.xfail(
    reason="Multiple values for a single-valued field are not yet supported; should pick one or raise a clear error"
)
def test_multiple_values_single_field(graph: Graph, EX: Namespace):
    """Test behavior when RDF has multiple values for a single-valued field."""

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX

        name: str  # Single-valued field

    # Create entity with multiple values for name
    d = Describer(graph=graph, about=EX.person1)
    d.rdftype(EX.Person)
    d.value(EX.name, Literal("John"))
    d.value(EX.name, Literal("Johnny"))  # Second value for same predicate

    # Expected behavior: should pick the first value or raise a clear error
    person = Person.parse_graph(graph, EX.person1)
    assert person.name in {"John", "Johnny"}


@pytest.mark.xfail(reason="Language-tagged literals are not yet supported; should handle language alternatives")
def test_language_tagged_literals(graph: Graph, EX: Namespace):
    """Test handling of language-tagged literals in RDF."""

    class MultiLingualContent(BaseRdfModel):
        rdf_type = EX.Content
        _rdf_namespace = EX

        title: str  # No language handling in basic field

    d = Describer(graph=graph, about=EX.content1)
    d.rdftype(EX.Content)
    # Add same title in different languages
    graph.add((EX.content1, EX.title, Literal("Hello", lang="en")))
    graph.add((EX.content1, EX.title, Literal("Bonjour", lang="fr")))

    # Expected behavior: should pick one value or provide language alternatives
    content = MultiLingualContent.parse_graph(graph, EX.content1)
    assert content.title in {"Hello", "Bonjour"}


@pytest.mark.xfail(reason="Blank nodes are not yet supported; should handle complex structures")
def test_blank_node_complex_structure(graph: Graph, EX: Namespace):
    """Test handling of complex blank node structures."""
    from rdflib import BNode

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

    # Create a person with address as blank node
    person_node = EX.person1
    address_node = BNode()

    # Add triples using blank node
    graph.add((person_node, EX.rdftype, EX.Person))
    graph.add((person_node, EX.name, Literal("John Doe")))
    graph.add((person_node, EX.address, address_node))
    graph.add((address_node, EX.rdftype, EX.Address))
    graph.add((address_node, EX.street, Literal("123 Main St")))
    graph.add((address_node, EX.city, Literal("Springfield")))

    # This should work - blank nodes should be handled
    person = Person.parse_graph(graph, EX.person1)
    assert person.name == "John Doe"
    assert person.address.street == "123 Main St"
    assert person.address.city == "Springfield"


def test_multiple_rdf_types(graph: Graph, EX: Namespace):
    """Test handling of resources with multiple rdf:types."""

    class Employee(BaseRdfModel):
        rdf_type = EX.Employee
        _rdf_namespace = EX
        name: str

    # Create entity with multiple types
    d = Describer(graph=graph, about=EX.person1)
    d.rdftype(EX.Employee)
    d.rdftype(EX.Manager)  # Additional type
    d.value(EX.name, Literal("John Doe"))

    # Current implementation might not handle this well
    employee = Employee.parse_graph(graph, EX.person1)
    assert employee.name == "John Doe"  # Should still work if one type matches


@pytest.mark.xfail(reason="RDF lists are not yet supported; should preserve order from rdf:List")
def test_rdf_list_ordering(graph: Graph, EX: Namespace):
    """Test handling of RDF ordered lists (rdf:List)."""
    from rdflib import RDF, URIRef

    class Playlist(BaseRdfModel):
        rdf_type = EX.Playlist
        _rdf_namespace = EX

        name: str
        tracks: list[str]  # Should preserve order from rdf:List

    # Create a playlist with ordered tracks using rdf:List
    playlist_uri = EX.playlist1
    graph.add((playlist_uri, EX.rdftype, EX.Playlist))
    graph.add((playlist_uri, EX.name, Literal("My Playlist")))

    # Create an RDF List for tracks
    track1 = URIRef("http://music.example.com/track1")
    track2 = URIRef("http://music.example.com/track2")
    track3 = URIRef("http://music.example.com/track3")

    # Create the list structure
    list_head = BNode()
    graph.add((playlist_uri, EX.tracks, list_head))

    current = list_head
    for track in [track1, track2, track3]:
        graph.add((current, RDF.first, track))
        if track != track3:
            next_node = BNode()
            graph.add((current, RDF.rest, next_node))
            current = next_node
        else:
            graph.add((current, RDF.rest, RDF.nil))

    # Expected behavior: tracks should be in order
    playlist = Playlist.parse_graph(graph, EX.playlist1)
    assert playlist.name == "My Playlist"
    assert playlist.tracks == [track1, track2, track3]


@pytest.mark.xfail(reason="RDF reification is not yet supported; should parse reified statements")
def test_reification_handling(graph: Graph, EX: Namespace):
    """Test handling of RDF reification (statements about statements)."""
    from rdflib import RDF

    class Statement(BaseRdfModel):
        rdf_type = RDF.Statement
        _rdf_namespace = EX

        subject: str
        predicate: str
        object: str
        confidence: float  # A property of the statement itself

    # Create a base statement
    base_statement = EX.statement1
    graph.add((base_statement, RDF.type, RDF.Statement))
    graph.add((base_statement, RDF.subject, EX.JohnDoe))
    graph.add((base_statement, RDF.predicate, EX.knows))
    graph.add((base_statement, RDF.object, EX.JaneSmith))
    graph.add((base_statement, EX.confidence, Literal(0.9)))

    # Expected behavior: should parse the reified statement
    statement = Statement.parse_graph(graph, base_statement)
    assert statement.subject == str(EX.JohnDoe)
    assert statement.predicate == str(EX.knows)
    assert statement.object == str(EX.JaneSmith)
    assert statement.confidence == 0.9


@pytest.mark.xfail(reason="Custom RDF datatypes are not yet supported; should parse custom datatypes")
def test_custom_datatype_literals(graph: Graph, EX: Namespace):
    """Test handling of RDF literals with custom datatypes."""
    import datetime

    from rdflib import XSD

    class CustomDatatypes(BaseRdfModel):
        rdf_type = EX.CustomDatatypes
        _rdf_namespace = EX

        coordinate: str  # Should be a custom geo:point datatype
        date: datetime.date  # Should be xsd:date

    # Create custom datatype URIs
    GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
    point_literal = Literal("45.123,-122.456", datatype=GEO.point)
    date_literal = Literal("2024-03-20", datatype=XSD.date)

    d = Describer(graph=graph, about=EX.data1)
    d.rdftype(EX.CustomDatatypes)
    graph.add((EX.data1, EX.coordinate, point_literal))
    graph.add((EX.data1, EX.date, date_literal))

    # Expected behavior: should parse custom datatypes as strings
    data = CustomDatatypes.parse_graph(graph, EX.data1)
    assert data.coordinate == "45.123,-122.456"
    assert data.date == datetime.date(2024, 3, 20)


@pytest.mark.xfail(reason="RDF container membership properties are not yet supported; should parse container items")
def test_container_membership(graph: Graph, EX: Namespace):
    """Test handling of RDF container membership properties."""
    from rdflib import RDF

    class Container(BaseRdfModel):
        rdf_type = RDF.Bag  # Using RDF Bag as an example
        _rdf_namespace = EX

        items: list[str]

    # Create a Bag container
    container_uri = EX.container1
    graph.add((container_uri, RDF.type, RDF.Bag))

    # Add items using container membership properties (_1, _2, _3, etc.)
    for i, item in enumerate(["item1", "item2", "item3"], start=1):
        member_prop = RDF[f"_{i}"]  # Creates properties like rdf:_1, rdf:_2, etc.
        graph.add((container_uri, member_prop, Literal(item)))

    # Expected behavior: should parse all items in order
    container = Container.parse_graph(graph, container_uri)
    assert container.items == ["item1", "item2", "item3"]


def test_datatype_promotion(graph: Graph, EX: Namespace):
    """Test handling of RDF literal datatype promotion/coercion."""

    class Numbers(BaseRdfModel):
        rdf_type = EX.Numbers
        _rdf_namespace = EX

        integer_field: int
        decimal_field: float
        any_number: float

    d = Describer(graph=graph, about=EX.numbers1)
    d.rdftype(EX.Numbers)

    # Add an integer that could be promoted to decimal
    graph.add((EX.numbers1, EX.integer_field, Literal("42", datatype=XSD.integer)))
    graph.add((EX.numbers1, EX.decimal_field, Literal("42.0", datatype=XSD.decimal)))
    # Add an integer to a float field - should be promoted
    graph.add((EX.numbers1, EX.any_number, Literal("42", datatype=XSD.integer)))

    numbers = Numbers.parse_graph(graph, EX.numbers1)
    assert isinstance(numbers.integer_field, int)
    assert isinstance(numbers.decimal_field, float)
    assert isinstance(numbers.any_number, float)
    assert numbers.any_number == 42.0


@pytest.mark.xfail(reason="Blank nodes are not yet supported")
def test_blank_node_identity(graph: Graph, EX: Namespace):
    """Test handling of blank node identity and equivalence."""

    class NodeContainer(BaseRdfModel):
        rdf_type = EX.NodeContainer
        _rdf_namespace = EX

        name: str
        related: Annotated[list["NodeContainer"], WithPredicate(EX.related)]

    NodeContainer.model_rebuild()

    # Create two containers that reference the same blank node
    container1_uri = EX.container1
    container2_uri = EX.container2
    shared_blank_node = BNode()

    # Add data about the blank node
    graph.add((shared_blank_node, RDF.type, EX.NodeContainer))
    graph.add((shared_blank_node, EX.name, Literal("Shared Node")))

    # Both containers relate to the same blank node
    graph.add((container1_uri, RDF.type, EX.NodeContainer))
    graph.add((container1_uri, EX.name, Literal("Container 1")))
    graph.add((container1_uri, EX.related, shared_blank_node))

    graph.add((container2_uri, RDF.type, EX.NodeContainer))
    graph.add((container2_uri, EX.name, Literal("Container 2")))
    graph.add((container2_uri, EX.related, shared_blank_node))

    # Parse both containers
    container1 = NodeContainer.parse_graph(graph, container1_uri)
    container2 = NodeContainer.parse_graph(graph, container2_uri)

    # The related blank nodes should have the same content but might not be the same object
    assert len(container1.related) == 1
    assert len(container2.related) == 1
    assert container1.related[0].name == "Shared Node"
    assert container2.related[0].name == "Shared Node"
    # This might fail because we create separate objects for the same blank node
    assert container1.related[0] is not container2.related[0]


@pytest.mark.xfail(reason="We don't support SPARQL-like property paths.")
def test_property_paths(graph: Graph, EX: Namespace):
    """Test handling of complex property path patterns."""

    class Organization(BaseRdfModel):
        rdf_type = EX.Organization
        _rdf_namespace = EX

        name: str
        # These would be nice to have but aren't supported
        all_employees: Annotated[list["Person"], WithPredicate(EX.department / EX.employee)]  # type: ignore # Path expression
        matrix_managers: Annotated[
            list["Person"], WithPredicate(EX.department / EX.manager | EX.project / EX.leader)  # type: ignore
        ]  # Alternative paths

    class Person(BaseRdfModel):
        rdf_type = EX.Person
        _rdf_namespace = EX
        name: str

    Organization.model_rebuild()
    Person.model_rebuild()

    # Create a complex organizational structure
    org = EX.org1
    dept1 = BNode()
    dept2 = BNode()
    proj1 = BNode()

    # Add basic org data
    graph.add((org, RDF.type, EX.Organization))
    graph.add((org, EX.name, Literal("Test Org")))

    # Add departments and employees
    graph.add((org, EX.department, dept1))
    graph.add((org, EX.department, dept2))
    graph.add((org, EX.project, proj1))

    # Add people
    for i, uri in enumerate([EX.person1, EX.person2, EX.person3]):
        graph.add((uri, RDF.type, EX.Person))
        graph.add((uri, EX.name, Literal(f"Person {i}")))

    # Add relationships
    graph.add((dept1, EX.employee, EX.person1))
    graph.add((dept2, EX.employee, EX.person2))
    graph.add((dept1, EX.manager, EX.person3))
    graph.add((proj1, EX.leader, EX.person2))

    # Try to parse with property paths
    org = Organization.parse_graph(graph, org)

    # Should find all employees through departments
    assert len(org.all_employees) == 2

    # Should find all managers/leaders through either departments or projects
    assert len(org.matrix_managers) == 2
