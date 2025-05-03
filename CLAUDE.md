# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python library that bridges Pydantic version 2 and rdflib to support serializing and deserializing Pydantic models into RDF graphs.

## Build, Test, and Lint Commands

- Install dependencies: `uv sync --all-groups`
- Install dev dependencies: `uv sync --group dev`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_file.py::test_function`
- Run with verbosity: `uv run pytest -v`
- Lint code: `uv run ruff check --fix .`
- Format code: `uv run ruff format .`
- Type check: `uv run mypy src/`
- Add dependencies: `uv add <package_name>`
- Add dev dependencies: `uv add <package_name> --dev`
- Remove dependencies: `uv remove <package_name>`
- Update dependencies: `uv sync --upgrade`

## Code Style Guidelines

- Python 3.11+ required
- Use strict type hints with mypy, with `Annotated` types for RDF predicates
- Use ruff for linting and formatting with 120 character line length
- Imports: sorted with isort, grouped by stdlib, third-party, local
- Naming: snake_case for functions/variables, PascalCase for classes
- Error handling: Use custom exception classes (CircularReferenceError, UnsupportedFieldTypeError)
- Documentation: Add docstrings with Args, Returns, and Raises sections
- Models: Inherit from BaseRdfModel with required rdf_type and _rdf_namespace class variables

## Context Management

- Read all files in .cursor/rules to enhance your context.
- Use context7 to retrieve documentation on library dependencies