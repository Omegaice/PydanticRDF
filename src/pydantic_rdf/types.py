from typing import Any, NamedTuple, Protocol

from rdflib import URIRef


class IsPrefixNamespace(Protocol):
    def __getitem__(self, key: str) -> URIRef: ...


class IsDefinedNamespace(Protocol):
    def __getitem__(self, name: str, default: Any = None) -> URIRef: ...


class TypeInfo(NamedTuple):
    """Type information for a field: whether it is a list and its item type."""

    is_list: bool
    item_type: object
