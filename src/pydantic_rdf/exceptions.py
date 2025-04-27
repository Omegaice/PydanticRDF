from rdflib import URIRef


class CircularReferenceError(Exception):
    """Raised when a circular reference is detected during RDF parsing."""

    def __init__(self, value: URIRef):
        message = f"Circular reference detected for {value}"
        super().__init__(message)


class UnsupportedFieldTypeError(Exception):
    """Raised when an unsupported field type is encountered during RDF parsing."""

    def __init__(self, field_type: object, field_name: str):
        message = f"Unsupported field type: {field_type} for field {field_name}"
        super().__init__(message)
