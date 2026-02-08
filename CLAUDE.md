# Skills Arena - Claude Code Instructions

## Project Overview

**Skills Arena** is an SDK for benchmarking and optimizing AI agent skill descriptions - essentially "SEO for agent skills." It answers the question: *"When an agent has multiple skills available, will it pick yours?"*

## Quick Reference

```python
from skills_arena import Arena, Config

# Evaluate a single skill
results = Arena().evaluate("./skill.md", task="web search")

# Compare skills head-to-head
results = Arena().compare(["./a.md", "./b.md"], task="web search")

# Battle royale with ELO rankings
results = Arena().battle_royale(skills, task="web search")
```

## Architecture

```
Arena (entry point)
├── Parser      → Parses skill files (.md, JSON, etc.)
├── Generator   → Creates test scenarios via LLM
├── Runner      → Executes scenarios against agent frameworks
└── Scorer      → Calculates metrics (selection rate, ELO)
```

## Key Patterns

### 1. Pydantic Models Everywhere
All data structures use Pydantic `BaseModel` for validation:
```python
from pydantic import BaseModel, Field

class Skill(BaseModel):
    name: str
    description: str = ""
```

### 2. Async-First with Sync Wrappers
Core methods are async, with sync wrappers for convenience:
```python
async def evaluate_async(self, ...) -> EvaluationResult:
    ...

def evaluate(self, ...) -> EvaluationResult:
    return asyncio.run(self.evaluate_async(...))
```

### 3. Abstract Base Classes for Extensibility
Each component has a base class for new implementations:
- `BaseParser` → `ClaudeCodeParser`, `OpenAIParser`, `MCPParser`
- `BaseGenerator` → `LLMGenerator`, `MockGenerator`
- `BaseAgent` → `ClaudeCodeAgent`, `MockAgent`

### 4. Mock Classes for Testing
Every component has a mock for testing without API calls:
```python
# Use MockGenerator instead of LLMGenerator
# Use MockAgent instead of ClaudeCodeAgent
```

## Directory Structure

```
src/skills_arena/
├── __init__.py      # Public exports
├── arena.py         # Main Arena class
├── config.py        # Config dataclass
├── models.py        # Core data models
├── exceptions.py    # Custom exceptions
├── parser/          # Skill format parsers
├── generator/       # Scenario generation
├── runner/          # Agent runners
├── scorer/          # Metrics & ELO
├── insights/        # AI analysis (future)
└── reporter/        # Output formatting (future)
```

## Development Guidelines

### Adding a New Parser
1. Create `parser/new_format.py`
2. Extend `BaseParser`
3. Implement `parse()`, `can_parse()`, `format` property
4. Add to `parser/__init__.py` exports
5. Add tests in `tests/test_parser.py`

### Adding a New Agent
1. Create `runner/new_agent.py`
2. Extend `BaseAgent`
3. Implement `select_skill()` and `close()`
4. Add to `runner/__init__.py` and `get_agent()` factory
5. Add tests in `tests/test_runner.py`

### Running Tests
```bash
# All tests
pytest

# Specific module
pytest tests/test_arena.py

# With coverage
pytest --cov=skills_arena
```

### Code Style
- Use `ruff` for linting
- Type hints required for public APIs
- Docstrings for all public classes/methods

## Important Files

| File | Purpose |
|------|---------|
| `arena.py` | Main entry point, orchestrates all components |
| `models.py` | Core data models (Skill, Scenario, Result types) |
| `config.py` | Configuration with validation |
| `parser/parser.py` | Unified parser with auto-detection |
| `runner/claude_code.py` | Claude Code SDK integration |
| `scorer/metrics.py` | Selection rate, accuracy calculations |
| `scorer/elo.py` | ELO rating system |

## Agent Frameworks

The SDK tests against real agent frameworks, not just raw LLM APIs:

| Agent | SDK | Status |
|-------|-----|--------|
| `claude-code` | claude-code-sdk | Primary |
| `mock` | (built-in) | Testing |

## Common Tasks

### Add support for a new skill format
```bash
# 1. Create the parser
touch src/skills_arena/parser/new_format.py

# 2. Look at existing parsers for reference
cat src/skills_arena/parser/claude_code.py
```

### Test with mock agents (no API needed)
```python
config = Config(agents=["mock"])
arena = Arena(config)
results = arena.evaluate("./skill.md", task="test")
```

### Debug scenario generation
```python
from skills_arena import LLMGenerator, Skill

generator = LLMGenerator()
scenarios = await generator.generate(
    task="web search",
    skills=[skill],
    count=10,
)
for s in scenarios:
    print(f"{s.prompt} -> {s.expected_skill}")
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...  # Required for claude-code
```

## Dependencies

Core:
- `claude-code-sdk>=0.0.25` - Claude Code agent framework
- `anthropic>=0.40.0` - Anthropic API
- `pydantic>=2.0.0` - Data validation

## Git Workflow

- Branch naming: `feat/description`, `bugfix/description`
- Keep commits focused and atomic
- Run tests before committing: `pytest`
