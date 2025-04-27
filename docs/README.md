# PydanticRDF Documentation

This directory contains the documentation for PydanticRDF, a library that bridges Pydantic V2 models and RDF graphs.

## Building the Documentation

The documentation is written in Markdown and can be built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

To build the documentation:

1. Install the required packages:

```bash
uv sync --group docs
```

2. Build the documentation:

```bash
mkdocs build
```

3. Or serve the documentation locally:

```bash
mkdocs serve
```

## Documentation Structure

- `index.md`: Overview and introduction
- `quickstart.md`: Quick start guide

## Contributing to Documentation

When contributing to the documentation, please follow these guidelines:

1. Write in clear, concise language
2. Use code examples to illustrate concepts
3. Keep the API reference up-to-date with the codebase
4. Add examples for new features