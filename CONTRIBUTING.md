# Contributing to prompt-gen-mcp

🎉 Thank you for your interest in contributing to `prompt-gen-mcp`! This document provides guidelines and information for contributors.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- API keys for testing (PromptGen, GROQ, Tavily)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/prompt-gen-mcp.git
   cd prompt-gen-mcp
   ```

2. **Install Dependencies**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. **Set Environment Variables**
   ```bash
   export PROMPTGEN_API_KEY="pg_sk_your_test_key"
   export GROQ_API_KEY="gsk_your_test_key"
   export TAVILY_API_KEY="tvly_your_test_key"
   ```

4. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

## 📋 Contribution Guidelines

### Code Style

- **Python**: Follow PEP 8 standards
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Use Google-style docstrings
- **Formatting**: Use `black` for code formatting

```python
def enhance_prompt(question: str, workspace_path: str) -> dict:
    """Enhance a user question with context and techniques.
    
    Args:
        question: The user's question to enhance
        workspace_path: Path to the code workspace
        
    Returns:
        Dictionary containing enhanced prompt data
        
    Raises:
        ValueError: If question is empty or invalid
    """
    pass
```

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add support for custom technique filtering
fix(scanner): resolve file encoding issues on Windows
docs(readme): update installation instructions
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

## 🔧 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write code following style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_enhancement.py

# Run with coverage
python -m pytest --cov=src tests/
```

### 4. Format Code
```bash
# Format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### 5. Commit and Push
```bash
git add .
git commit -m "feat(scope): your descriptive message"
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Use the PR template
- Link related issues
- Add screenshots for UI changes
- Request review from maintainers

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/
│   ├── test_scanner.py
│   ├── test_enhancer.py
│   └── test_api_client.py
├── integration/
│   ├── test_full_pipeline.py
│   └── test_mcp_server.py
└── fixtures/
    ├── sample_code/
    └── mock_responses/
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from src.prompt_gen_mcp.enhancer import PromptEnhancer

class TestPromptEnhancer:
    def test_enhance_simple_question(self):
        """Test enhancement of a simple coding question."""
        enhancer = PromptEnhancer()
        result = enhancer.enhance("How do I optimize this function?")
        
        assert result["enhanced_prompt"] is not None
        assert len(result["techniques"]) > 0
        assert result["confidence_score"] > 0.5
    
    @patch('src.prompt_gen_mcp.api_client.PromptGenAPIClient')
    def test_api_failure_handling(self, mock_client):
        """Test graceful handling of API failures."""
        mock_client.fetch_techniques.side_effect = Exception("API Error")
        
        enhancer = PromptEnhancer()
        result = enhancer.enhance("Test question")
        
        # Should fallback gracefully
        assert result["status"] == "fallback"
        assert "error" in result
```

### Test Data

- Use fixtures for reusable test data
- Mock external API calls
- Never commit real API keys in tests
- Use temporary directories for file operations

## 📚 Documentation

### README Updates
- Keep installation instructions current
- Update feature lists
- Add new configuration options
- Include relevant examples

### Code Documentation
- Document all public functions
- Include usage examples
- Explain complex algorithms
- Add type hints everywhere

### API Documentation
- Document all MCP tools
- Include parameter descriptions
- Provide example requests/responses
- Update schema definitions

## 🔒 Security Considerations

### Code Review Checklist
- [ ] No hardcoded API keys or secrets
- [ ] Proper input validation
- [ ] Secure file handling
- [ ] No code content in API requests
- [ ] Error messages don't leak sensitive info

### Testing Security
- Test with invalid/malicious inputs
- Verify API key handling
- Check file permission handling
- Validate network request content

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues
2. Test with latest version
3. Verify configuration
4. Try minimal reproduction

### Bug Report Template
```markdown
**Bug Description**
Clear description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.11.0]
- Version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 🏆 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page
- Special mentions for significant contributions

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Discord**: [Community chat link]
- **Email**: [maintainer@promptgen.dev]

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to prompt-gen-mcp! 🚀**