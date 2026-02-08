# Skills Arena - Architecture

> **The SEO platform for AI agent skills** - Benchmark, optimize, and compete your skill descriptions.

## Vision

Just as SEO tools help websites rank higher in search engines, Skills Arena helps skill descriptions "rank higher" in AI agent decision-making. When an agent decides which tool/skill to use, **will it pick yours?**

---

## Design Principles

1. **SDK-First**: Python library with intuitive API - not a CLI tool with SDK bolted on
2. **Sensible Defaults**: Works out of the box with minimal config
3. **Progressive Disclosure**: Simple for basic use, powerful when needed
4. **Developer Experience**: Clear errors, good docs, fast feedback loop

---

## Quick Start (Target UX)

### Simplest Use Case

```python
from skills_arena import Arena

# One line to evaluate a skill
results = Arena().evaluate("./my-skill.md", task="web search")
print(results.score)  # 78.5
```

### Compare Skills

```python
from skills_arena import Arena

arena = Arena()
results = arena.compare(
    skills=["./my-skill.md", "./tavily-search.md"],
    task="web search and content extraction",
)

print(results.winner)           # "tavily-search"
print(results.selection_rates)  # {"my-skill": 0.42, "tavily-search": 0.58}
print(results.insights[0])      # "Your skill lacks specific examples for..."
```

### With Configuration

```python
from skills_arena import Arena, Config

config = Config(
    scenarios=100,                    # Number of test scenarios
    agents=["claude", "gpt-4o"],      # Frameworks to test
    temperature=0.7,                  # Scenario generation diversity
    include_adversarial=True,         # Edge cases
)

arena = Arena(config)
results = arena.battle_royale(
    skills=["./skill-a.md", "./skill-b.md", "./skill-c.md"],
    task="data analysis and visualization",
)

print(results.leaderboard)
# 1. skill-b (ELO: 1523)
# 2. skill-a (ELO: 1489)
# 3. skill-c (ELO: 1388)
```

### From Config File

```python
from skills_arena import Arena

# Load everything from YAML
arena = Arena.from_config("./arena.yaml")
results = arena.run()
```

```yaml
# arena.yaml
task: "web search and information retrieval"

skills:
  - path: ./skills/my-search.md
    name: my-search
  - path: ./skills/competitor.md
    name: competitor

evaluation:
  scenarios: 50
  agents: [claude-sonnet, gpt-4o]
  mode: compare  # evaluate, compare, battle_royale

output:
  format: json
  path: ./results/
```

---

## Core Concepts

### 1. Skill

A skill definition that an agent can choose to use. Parsed from various formats.

```python
@dataclass
class Skill:
    name: str
    description: str
    parameters: List[Parameter]
    when_to_use: List[str]      # Examples of triggering prompts
    source_format: str          # claude_code, openai, mcp, generic
    token_count: int
    raw_content: str            # Original file content
```

### 2. Task

A high-level description of what the skills are meant to do. Used to generate scenarios.

```python
# Simple string
task = "web search"

# Or detailed
task = Task(
    description="web search and information retrieval",
    domains=["enterprise", "developer", "casual"],  # Optional
    edge_cases=["rate limiting", "auth required"],  # Optional
)
```

### 3. Scenario

An auto-generated test prompt that should trigger skill selection.

```python
@dataclass
class Scenario:
    id: str
    prompt: str                 # The user message
    expected_skill: str         # Which skill SHOULD be chosen
    difficulty: str             # easy, medium, hard
    tags: List[str]             # categorization
```

### 4. Result

The outcome of an evaluation.

```python
@dataclass
class EvaluationResult:
    skill: str
    score: float                       # 0-100
    grade: str                         # A+, A, B+, etc.
    selection_rate: float              # 0.0-1.0
    false_positive_rate: float
    invocation_accuracy: float
    per_agent: Dict[str, AgentResult]  # Breakdown by framework
    insights: List[Insight]
    scenarios_run: int

@dataclass
class ComparisonResult:
    winner: str
    selection_rates: Dict[str, float]
    head_to_head: Dict[str, Dict[str, int]]  # skill -> opponent -> wins
    insights: List[Insight]
    per_agent: Dict[str, ComparisonResult]

@dataclass
class BattleResult:
    leaderboard: List[RankedSkill]
    elo_ratings: Dict[str, int]
    matchups: List[Matchup]
    insights: List[Insight]
```

---

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SKILLS ARENA SDK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Arena (Main Entry Point)                                                   │
│   ├── evaluate(skill, task) → EvaluationResult                              │
│   ├── compare(skills, task) → ComparisonResult                              │
│   ├── battle_royale(skills, task) → BattleResult                            │
│   └── insights(skill) → List[Insight]                                       │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│   │    Parser    │   │  Generator   │   │    Runner    │   │   Scorer    │ │
│   │              │   │              │   │              │   │             │ │
│   │ • claude_code│   │ • from_task  │   │ • claude     │   │ • metrics   │ │
│   │ • openai     │   │ • from_skill │   │ • openai     │   │ • elo       │ │
│   │ • mcp        │   │ • adversarial│   │ • gemini     │   │ • ranking   │ │
│   │ • generic    │   │              │   │ • mock       │   │             │ │
│   └──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘ │
│                                                                              │
│   ┌──────────────┐   ┌──────────────┐                                       │
│   │   Insights   │   │   Reporter   │                                       │
│   │              │   │              │                                       │
│   │ • why_lost   │   │ • dict/json  │                                       │
│   │ • optimize   │   │ • markdown   │                                       │
│   │ • compare    │   │ • dataframe  │                                       │
│   └──────────────┘   └──────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parser

Reads skill files and normalizes them to `Skill` objects.

```python
class Parser:
    @staticmethod
    def parse(path: str) -> Skill:
        """Auto-detect format and parse"""

    @staticmethod
    def parse_claude_code(path: str) -> Skill:
        """Parse Claude Code .md skill format"""

    @staticmethod
    def parse_openai(schema: dict) -> Skill:
        """Parse OpenAI function schema"""

    @staticmethod
    def parse_mcp(definition: dict) -> Skill:
        """Parse MCP tool definition"""
```

### Generator

Creates test scenarios from task descriptions.

```python
class Generator:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def generate(
        self,
        task: str,
        skills: List[Skill],
        count: int = 50,
        include_adversarial: bool = True,
    ) -> List[Scenario]:
        """
        Generate scenarios that should trigger skill selection.

        Process:
        1. Analyze task description
        2. Analyze each skill's "when to use" patterns
        3. Generate diverse prompts that should trigger each skill
        4. Add adversarial/edge cases
        5. Shuffle and return
        """
```

**Generation Prompt Strategy:**

```
Given the task "{task}" and these skills:

{skill_descriptions}

Generate {count} diverse user prompts that would require using these skills.
For each prompt, indicate which skill SHOULD be selected.

Requirements:
- Cover different user personas (developer, business user, casual)
- Include simple and complex requests
- Include edge cases and ambiguous scenarios
- Vary the phrasing and vocabulary
- {adversarial_instructions if include_adversarial}
```

### Runner

Executes scenarios against **agent frameworks** (not raw LLM APIs).

> **Important Distinction**: We test against real agent frameworks like Claude Code and Codex,
> not just raw LLM APIs with tool_use. This is critical because agent frameworks have their
> own skill/tool selection logic, system prompts, and behaviors that affect which skill gets chosen.

#### Supported Agent Frameworks

| Framework | SDK | Language | Status | Notes |
|-----------|-----|----------|--------|-------|
| **Claude Code** | `claude-code-sdk` | Python | ✅ Primary | Official Anthropic agent framework |
| **Codex** | `@openai/codex` | TypeScript only | ⚠️ Limited | Requires TS subprocess bridge |

#### Claude Code SDK Integration

Claude Code is the primary target agent. We use the official Claude Code SDK (Python).

```python
from claude_code_sdk import Claude, Message

class ClaudeCodeAgent(BaseAgent):
    """
    Uses Claude Code SDK to test skill selection.

    This spawns an actual Claude Code session with skills loaded,
    simulating real-world skill selection behavior.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def select_skill(
        self,
        prompt: str,
        available_skills: List[Skill],
    ) -> SkillSelection:
        """
        Run prompt through Claude Code with skills available.
        Observe which skill (if any) gets invoked.
        """
        # Convert skills to Claude Code skill format
        skill_definitions = [self._to_claude_skill(s) for s in available_skills]

        # Create Claude Code session with skills
        async with Claude(
            model=self.model,
            skills=skill_definitions,
        ) as claude:
            response = await claude.send(Message(content=prompt))

            # Extract which skill was selected (if any)
            selected_skill = self._extract_skill_selection(response)

            return SkillSelection(
                skill=selected_skill,
                confidence=response.confidence,
                reasoning=response.reasoning,
            )
```

#### Codex SDK Integration (TypeScript Limitation)

**Problem**: OpenAI's Codex SDK is **TypeScript-only**. Since Skills Arena is a Python SDK,
we need a bridge solution.

**Workaround Options**:

1. **Subprocess Bridge** (Recommended for MVP)
   ```python
   class CodexAgent(BaseAgent):
       """
       Bridges to Codex via TypeScript subprocess.
       Requires Node.js and @openai/codex installed.
       """

       async def select_skill(
           self,
           prompt: str,
           available_skills: List[Skill],
       ) -> SkillSelection:
           # Write skills and prompt to temp files
           config = self._prepare_codex_config(prompt, available_skills)

           # Run TypeScript bridge script
           result = await asyncio.create_subprocess_exec(
               "npx", "ts-node", "codex_bridge.ts",
               "--config", config_path,
               stdout=asyncio.subprocess.PIPE,
           )
           stdout, _ = await result.communicate()

           return self._parse_codex_result(stdout)
   ```

2. **HTTP Bridge** (For production)
   - Run a small Express server that wraps Codex
   - Python calls HTTP endpoints
   - More overhead but cleaner separation

3. **Skip Codex** (Simplest)
   - Focus on Claude Code for MVP
   - Add Codex support in Phase 2
   - Use raw OpenAI API as fallback

**Recommendation**: Start with Claude Code SDK only for MVP. Add Codex via subprocess bridge in Phase 2.

```python
class Runner:
    def __init__(self, agents: List[str] = ["claude-code"]):
        self.agents = [self._get_agent(a) for a in agents]

    def _get_agent(self, name: str) -> BaseAgent:
        agents = {
            "claude-code": ClaudeCodeAgent,      # Claude Code SDK
            "codex": CodexAgent,                  # Codex via TS bridge
            "mock": MockAgent,                    # For testing
        }
        return agents[name]()

    async def run(
        self,
        scenarios: List[Scenario],
        skills: List[Skill],
    ) -> List[SelectionResult]:
        """
        Run scenarios through agent frameworks and record which skill they choose.
        """

class BaseAgent(ABC):
    @abstractmethod
    async def select_skill(
        self,
        prompt: str,
        available_skills: List[Skill],
    ) -> SkillSelection:
        """Given a prompt and skills, which skill would you use?"""

class MockAgent(BaseAgent):
    """For testing - deterministic responses"""
```

### Scorer

Calculates metrics from run results.

```python
class Scorer:
    @staticmethod
    def score_evaluation(
        skill: Skill,
        results: List[SelectionResult],
    ) -> EvaluationResult:
        """Calculate metrics for single skill evaluation"""

    @staticmethod
    def score_comparison(
        skills: List[Skill],
        results: List[SelectionResult],
    ) -> ComparisonResult:
        """Calculate head-to-head comparison metrics"""

    @staticmethod
    def score_battle(
        skills: List[Skill],
        results: List[SelectionResult],
    ) -> BattleResult:
        """Calculate ELO rankings and leaderboard"""

class ELO:
    """ELO rating system for skill rankings"""

    @staticmethod
    def update(winner_elo: int, loser_elo: int, k: int = 32) -> Tuple[int, int]:
        """Update ELO ratings after a matchup"""
```

### Insights

AI-powered analysis and recommendations.

```python
class InsightsEngine:
    async def analyze(
        self,
        skill: Skill,
        results: EvaluationResult | ComparisonResult,
        competitors: List[Skill] = None,
    ) -> List[Insight]:
        """Generate actionable insights"""

    async def suggest_improvements(
        self,
        skill: Skill,
        results: EvaluationResult,
    ) -> List[Suggestion]:
        """Suggest concrete description improvements"""

    async def explain_losses(
        self,
        skill: Skill,
        lost_scenarios: List[SelectionResult],
    ) -> List[Explanation]:
        """Explain why skill lost specific scenarios"""
```

### Reporter

Format results for output.

```python
class Reporter:
    @staticmethod
    def to_dict(result: Result) -> dict:
        """Convert to dictionary"""

    @staticmethod
    def to_json(result: Result, path: str = None) -> str:
        """Convert to JSON string or write to file"""

    @staticmethod
    def to_markdown(result: Result) -> str:
        """Generate markdown report"""

    @staticmethod
    def to_dataframe(result: Result) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
```

---

## Configuration

### Config Object

```python
@dataclass
class Config:
    # Scenario generation
    scenarios: int = 50
    generator_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    include_adversarial: bool = True

    # Agent frameworks to test against
    # Options: "claude-code", "codex", "mock"
    agents: List[str] = field(default_factory=lambda: ["claude-code"])

    # Execution
    parallel_requests: int = 10
    timeout_seconds: int = 30

    # Scoring
    elo_k_factor: int = 32

    # Output
    verbose: bool = False

    # API Keys (defaults to env vars)
    anthropic_api_key: str = None
    openai_api_key: str = None

    # Codex bridge settings (if using codex agent)
    codex_bridge_path: str = None  # Path to TS bridge script
    node_executable: str = "node"  # Node.js path
```

### YAML Config

```yaml
# arena.yaml
task: "web search and information retrieval"

skills:
  - ./skills/my-search.md
  - ./skills/competitor.md
  # Or with names
  - path: ./skills/another.md
    name: custom-name

evaluation:
  scenarios: 100
  agents:
    - claude-sonnet
    - gpt-4o
  include_adversarial: true

output:
  format: json
  path: ./results/
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional: default config
SKILLS_ARENA_CONFIG=./arena.yaml
```

---

## Agent Framework Dependencies

Different agent frameworks have different requirements. Here's what you need for each:

### Claude Code (Primary - Recommended)

**Requirements:**
- Python 3.10+
- `claude-code-sdk` package
- Anthropic API key

```bash
pip install skills-arena  # Includes claude-code-sdk
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
# Just works
arena = Arena(Config(agents=["claude-code"]))
```

### Codex (TypeScript Limitation)

**Problem**: Codex SDK (`@openai/codex`) is TypeScript-only. We provide a bridge.

**Requirements:**
- Node.js 18+
- `@openai/codex` npm package
- OpenAI API key

```bash
# Install Node.js dependencies (one-time)
npm install -g @openai/codex typescript ts-node

# Set API key
export OPENAI_API_KEY=sk-...

# Initialize the bridge (creates codex_bridge.ts)
skills-arena init-codex
```

```python
# Now codex works via subprocess bridge
arena = Arena(Config(agents=["codex"]))
```

**How the Bridge Works:**

```
Python (skills-arena) ──JSON──▶ subprocess ──▶ TypeScript (codex_bridge.ts)
                                                      │
                                                      ▼
                                              Codex SDK (@openai/codex)
                                                      │
◀──JSON── subprocess ◀── result ◀────────────────────┘
```

**Limitations:**
- Adds ~100-200ms latency per call (subprocess overhead)
- Requires Node.js installed
- More complex debugging

### Comparison Matrix

| Agent | Language | Setup | Latency | Realism |
|-------|----------|-------|---------|---------|
| `claude-code` | Python | Easy | Fast | High (real agent) |
| `codex` | TS→Python bridge | Medium | +100-200ms | High (real agent) |

**Recommendation for MVP**: Start with `claude-code` only. Add `codex` in Phase 2.

---

## Error Handling & UX

### Clear Error Messages

```python
# Bad:
# raise ValueError("Invalid skill")

# Good:
raise SkillParseError(
    f"Could not parse skill at '{path}'.\n"
    f"Expected Claude Code format (.md with description section).\n"
    f"See: https://skills-arena.dev/docs/formats"
)
```

### Progress Feedback

```python
arena = Arena(config)
results = arena.compare(
    skills=skills,
    task=task,
    on_progress=lambda p: print(f"{p.stage}: {p.percent}%")
)

# Output:
# Parsing skills: 100%
# Generating scenarios: 100%
# Running evaluations: 45%
# ...
```

### Async Support

```python
# Sync (blocking)
results = arena.evaluate(skill, task)

# Async
results = await arena.evaluate_async(skill, task)

# With asyncio
import asyncio
results = asyncio.run(arena.evaluate_async(skill, task))
```

---

## Directory Structure

```
skills-arena/
├── src/
│   └── skills_arena/
│       ├── __init__.py          # Public API exports
│       ├── arena.py             # Main Arena class
│       ├── config.py            # Config dataclass
│       ├── models.py            # Skill, Scenario, Result dataclasses
│       ├── exceptions.py        # Custom exceptions
│       │
│       ├── parser/
│       │   ├── __init__.py
│       │   ├── base.py          # Parser interface
│       │   ├── claude_code.py   # Claude Code .md parser
│       │   ├── openai.py        # OpenAI function schema
│       │   ├── mcp.py           # MCP tool definition
│       │   └── generic.py       # Plain text fallback
│       │
│       ├── generator/
│       │   ├── __init__.py
│       │   ├── base.py          # Generator interface
│       │   └── llm.py           # LLM-based scenario generation
│       │
│       ├── runner/
│       │   ├── __init__.py
│       │   ├── base.py          # Agent interface
│       │   ├── claude_code.py   # Claude Code SDK agent
│       │   ├── codex.py         # Codex via TS bridge
│       │   └── mock.py          # Testing
│       │
│       ├── bridges/
│       │   ├── __init__.py
│       │   └── codex_bridge.ts  # TypeScript bridge for Codex
│       │
│       ├── scorer/
│       │   ├── __init__.py
│       │   ├── metrics.py       # Core metrics
│       │   └── elo.py           # ELO rating system
│       │
│       ├── insights/
│       │   ├── __init__.py
│       │   └── engine.py        # AI insights
│       │
│       └── reporter/
│           ├── __init__.py
│           ├── json.py
│           ├── markdown.py
│           └── dataframe.py
│
├── tests/
│   ├── test_arena.py
│   ├── test_parser.py
│   ├── test_generator.py
│   └── fixtures/
│       └── skills/              # Test skill files
│
├── examples/
│   ├── basic_evaluation.py
│   ├── compare_skills.py
│   ├── battle_royale.py
│   └── sample_skills/
│
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
└── LICENSE
```

---

## pyproject.toml

```toml
[project]
name = "skills-arena"
version = "0.1.0"
description = "Benchmark and optimize AI agent skill descriptions"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Eyal Ben Barouch", email = "eyal@tavily.com"}
]
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["ai", "agents", "skills", "benchmarking", "llm", "tools"]

dependencies = [
    "claude-code-sdk>=0.1.0",  # Claude Code agent framework
    "anthropic>=0.40.0",       # Anthropic API
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "httpx>=0.27.0",
    "tenacity>=8.0.0",         # Retries
]

[project.optional-dependencies]
codex = [
    # Codex requires Node.js + @openai/codex installed separately
    # This just documents the dependency
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.10.0",
]
pandas = [
    "pandas>=2.0.0",
]
all = [
    "skills-arena[dev,pandas]",
]

[project.urls]
Homepage = "https://github.com/eyalbenba/skills-arena"
Repository = "https://github.com/eyalbenba/skills-arena"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/skills_arena"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Feature Roadmap

### Phase 1: MVP Core
- [ ] `Arena.evaluate()` - single skill evaluation
- [ ] `Arena.compare()` - two-skill comparison
- [ ] Claude Code skill parser
- [ ] LLM scenario generator (from task description)
- [ ] Claude agent runner
- [ ] Basic metrics (selection rate, accuracy)
- [ ] Dict/JSON output
- [ ] pyproject.toml + pip installable

### Phase 2: Multi-Framework
- [ ] OpenAI agent runner
- [ ] Gemini agent runner
- [ ] OpenAI function schema parser
- [ ] MCP tool definition parser
- [ ] Cross-framework comparison

### Phase 3: Battle & Rankings
- [ ] `Arena.battle_royale()` - multi-skill competition
- [ ] ELO rating system
- [ ] Leaderboard generation
- [ ] Markdown reports

### Phase 4: Insights
- [ ] AI insights engine
- [ ] "Why you lost" analysis
- [ ] Optimization suggestions
- [ ] Suggested rewrites

### Phase 5: Polish & Ecosystem
- [ ] CLI wrapper (optional)
- [ ] Web UI (optional)
- [ ] GitHub Action for CI
- [ ] Documentation site

### Phase 6: skills.sh Integration
Integrate with the [skills.sh](https://skills.sh) public skill registry to enable:

- [ ] **Browse skills** - Search and discover skills via `npx skills find [query]`
- [ ] **Compare from registry** - Pull two skills and benchmark head-to-head
- [ ] **Publish scores** - Contribute benchmark results back to registry
- [ ] **Leaderboards** - Public skill rankings by task category

```python
# Future API concept
from skills_arena import Arena

arena = Arena()

# Browse skills for a task
skills = arena.browse("web search")  # Fetches from skills.sh registry

# Compare top 2 from registry
results = arena.compare(
    skills=["vercel-labs/skills/tavily-search", "openai/skills/web-browse"],
    task="web search",
)
```

---

## Open Questions

1. **Ground Truth Validation**: How do we verify generated scenarios are actually correct?
   - Current approach: Self-consistency (scenario generated FOR skill X → X is expected)
   - Alternative: LLM judge validates each scenario
   - Alternative: Human spot-check sample

2. **Agent Prompting**: How do we ask agents to "choose a skill"?
   - Option A: Present skills as tools, see which tool is called
   - Option B: Ask directly "which skill would you use?"
   - Option C: Both, compare results

3. **Skill Format Detection**: Auto-detect vs explicit format specification?
   - Current: Auto-detect from file extension + content
   - Alternative: Require explicit format in config
