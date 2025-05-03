# Changelog

All notable changes to the pydantic-rdf project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-05-02

### Added
- Support for JSON schema generation with proper handling of URIRef fields
- New `PydanticURIRef` type using Pydantic's annotation system
- Tests for schema generation and validation

### Changed
- Updated `BaseRdfModel` to use `PydanticURIRef` instead of raw `URIRef` for better schema support
- Improved type checking across the library

## [0.1.0] - Initial Release

### Added
- Initial implementation of `BaseRdfModel` for RDF graph integration
- Support for serializing Pydantic models to RDF graphs
- Support for deserializing RDF graphs to Pydantic models
- Field annotations (`WithPredicate`, `WithDataType`) for custom RDF mapping
- Support for nested models and list fields
- Comprehensive test suite