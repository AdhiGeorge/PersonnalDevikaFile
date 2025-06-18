# Contributing to AgentRes

Thank you for your interest in contributing to AgentRes! We welcome all contributions, whether they're bug reports, feature requests, documentation improvements, or code contributions.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/your-username/AgentRes.git
   cd AgentRes
   ```
3. **Set up** the development environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests
   ```bash
   pytest
   ```
4. Commit your changes with a descriptive message
   ```bash
   git commit -m "Add feature: your feature description"
   ```
5. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a pull request against the `main` branch

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Include docstrings for all public functions and classes
- Keep lines under 100 characters when possible

## Testing

- Write tests for all new features and bug fixes
- Run all tests before submitting a pull request
- Make sure existing tests pass

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Any relevant error messages or logs

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
