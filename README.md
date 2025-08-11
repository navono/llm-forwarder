# Python Project Template

This is a template project for Python applications. It provides a basic structure for creating Python applications with common configurations and tools.

## Features

- Modern Python project structure
- FastAPI HTTP server setup
- Logging configuration with loguru
- Configuration management
- Makefile for common tasks
- Ruff for code linting

## Project Structure

```
├── config/           # Configuration files
├── logs/             # Log files
├── scripts/          # Utility scripts
├── src/              # Source code
│   └── llm_forwarder/  # Main package
│       ├── http/     # HTTP server components
│       └── utils/    # Utility modules
├── .gitignore        # Git ignore file
├── Makefile          # Makefile for common tasks
├── pyproject.toml    # Project configuration
└── README.md         # This file
```

## Getting Started

### Prerequisites

- Python 3.12 or higher
- uv (Python package manager)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   uv pip install -e .
   ```

### Running the Application

```bash
make dev
```

## Using as a Template

This project can be used as a template for new Python projects. The template includes a script to rename all occurrences of the template name to your new project name.

### Creating a New Project from Template

To create a new project from this template, use the following command:

```bash
make create-project name="your-new-project-name"
```

This will:
1. Replace all occurrences of "llm-forwarder" with your new project name
2. Replace all occurrences of "llm_forwarder" with the snake_case version of your project name
3. Rename directories and files containing the template name

#### Dry Run Mode

If you want to see what changes would be made without actually making them, use the dry-run version:

```bash
make create-project-dry-run name="your-new-project-name"
```

This will show all files and directories that would be modified or renamed, but won't make any actual changes.

### Example

```bash
make create-project name="awesome-app"
```

This will convert the project template to "awesome-app" with the package name "awesome_app".

## Development

### Code Linting

```bash
make ruff-fix
```
