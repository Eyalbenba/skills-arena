<p align="center">
  <img src=".assets/skillsarena_logo.png" alt="Skills Arena" width="200">
</p>

<p align="center">
  <strong>Is your skill winning the context window?</strong>
</p>

<p align="center">
  <a href="https://skillsarena.ai"><img src="https://img.shields.io/badge/Website-skillsarena.ai-7ed957?style=flat&labelColor=1a1a1a&logo=google-chrome&logoColor=white" alt="Website"></a>
  <a href="https://pypi.org/project/skills-arena/"><img src="https://img.shields.io/pypi/v/skills-arena?style=flat&color=7ed957&labelColor=1a1a1a&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/Eyalbenba/skills-arena/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat&labelColor=1a1a1a" alt="License"></a>
  <a href="https://github.com/Eyalbenba/skills-arena"><img src="https://img.shields.io/github/stars/Eyalbenba/skills-arena?style=flat&labelColor=1a1a1a&color=yellow" alt="GitHub Stars"></a>
</p>

<p align="center">
  <a href="#why-skills-arena">Why?</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#custom-scenarios">Custom Scenarios</a> •
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/f90454dc-d432-4039-ae1f-aa15f8af198c" width="800" autoplay loop muted playsinline></video>
</p>

---

## Why Skills Arena?

Your skill's description is the most important copy you'll ever write. It's read by coding agents thousands of times a day, and it determines whether your product gets used or ignored.

**Skills are the new SEO.** Just like you used to optimize for Google's algorithm, you now need to optimize for the coding agent's decision-making process. Every day, thousands of decisions happen inside context windows — your skill vs. competitors, your description vs. theirs. **And you have no idea who's winning.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Developer: "Find the latest AI news and summarize the key points"     │
│                                                                         │
│  Coding Agent's Context Window:                                         │
│    • Your Search Skill                                                  │
│    • Competitor's Web Scraper                                           │
│    • Built-in WebSearch                                                 │
│                                                                         │
│  ⚡ One satisfies the request. The rest are forgotten.                  │
│  📊 Skills Arena shows you who wins — and why.                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Skills Arena is a **skill discovery optimization** framework — benchmark and improve how coding agents find and choose your skill.

> *You spent months building a great product. You wrote a skill so agents can use it. But right now, inside thousands of terminals, a coding agent is reading your skill description next to your competitor's — and choosing theirs. You never even knew it happened.*

## How It Works

```
                            ┌──────────────────────────────────────────────────────────┐
                            │              S C E N A R I O   G E N E R A T I O N       │
 ┌─────────────────┐        │                                                          │
 │   YOUR SKILL    │───────▶│   "Store embeddings"       → should pick: Your Skill    │
 │  vector-db.md   │        │   "Semantic search docs"   → should pick: Your Skill    │
 └─────────────────┘        │   "Scale to 1B vectors"    → should pick: Your Skill    │
                            │                                                          │
 ┌─────────────────┐        │   "Hybrid keyword+vector"  → should pick: Competitor    │
 │   COMPETITOR    │───────▶│   "Filter by metadata"     → should pick: Competitor    │
 │  rival-db.md    │        │                                                          │
 └─────────────────┘        └────────────────────────────┬─────────────────────────────┘
                                                         │
                                                         ▼
                            ┌──────────────────────────────────────────────────────────┐
                            │              A G E N T   S I M U L A T I O N             │
                            │                                                          │
                            │   Agent sees ALL skills in context, picks ONE per task  │
                            │                                                          │
                            │   ┌─────────────────────────────────────────────────┐    │
                            │   │ "Store embeddings"                              │    │
                            │   │  Expected: Your Skill                           │    │
                            │   │  Agent picked: Your Skill ✅ WIN                │    │
                            │   └─────────────────────────────────────────────────┘    │
                            │   ┌─────────────────────────────────────────────────┐    │
                            │   │ "Semantic search docs"                          │    │
                            │   │  Expected: Your Skill                           │    │
                            │   │  Agent picked: Competitor 🔴 STOLEN!            │    │
                            │   └─────────────────────────────────────────────────┘    │
                            └────────────────────────────┬─────────────────────────────┘
                                                         │
                                                         ▼
                            ┌──────────────────────────────────────────────────────────┐
                            │                    R E S U L T S                         │
                            │                                                          │
                            │   Your Skill        ████████████░░░░░░   60% selected    │
                            │   Competitor        ████████░░░░░░░░░░   40% selected    │
                            │                                                          │
                            │   🔴 STEALS: Competitor won 2 of your scenarios          │
                            │   🏆 WINNER: Your Skill (but watch those steals!)        │
                            └──────────────────────────────────────────────────────────┘
```

**The flow:**
1. **Input skills** — yours and the competition
2. **Generate scenarios** — prompts where each skill *should* be chosen
3. **Simulate** — a real agent sees all skills and picks one per task
4. **Track** — wins, losses, and steals (when competitors take *your* scenarios)
5. **Report** — selection rates, reasoning, and actionable insights

## Quick Start

### Installation

```bash
pip install skills-arena
```

### Compare Two Skills

```python
from skills_arena import Arena, Config

arena = Arena()
results = arena.compare(
    skills=["./my-skill.md", "./competitor.md"],
    task="web search and content extraction",
)

print(f"Winner: {results.winner}")
print(f"Selection rates: {results.selection_rates}")
```

**Output:**
```
======================================================================
RESULTS
======================================================================

🏆 Winner: Competitor Skill

📊 Selection Rates:
  My Skill             ██████               30%
  Competitor Skill     ██████████████       70%

📋 Scenarios run: 10

----------------------------------------------------------------------
🔴 STEAL DETECTION
----------------------------------------------------------------------
  My Skill: Lost 2 scenario(s) to competitors
```

### Optimize a Skill

Lost the comparison? Let the optimizer fix it:

```python
result = arena.optimize(
    skill="./my-skill.md",
    competitors=["./competitor.md"],
    task="web search and content extraction",
    max_iterations=2,
)

print_results(result)
```

**Output:**
```
======================================================================
OPTIMIZATION RESULTS
======================================================================
Skill:       My Skill
Competitors: Competitor Skill
Scenarios:   6  |  Iterations: 2

Before -> After:
  Selection Rate:  ███████░░░░░░░░░░░░░ 33%  ->  █████████████░░░░░░░ 67%  (+34%)
  Grade:             F  ->  D
  Tokens:           43  ->  40  (-3)

----------------------------------------------------------------------
Iteration 1:  33% -> 67%  (+34%)  [improved]

  Added concrete usage examples, specified output format,
  and differentiated from scraping tools.

  Scenarios:  3 won  |  0 stolen
```

The optimizer runs a **compare → rewrite → verify** loop:
1. Baseline comparison to measure current performance
2. LLM rewrites the description using competition data and stolen scenario reasoning
3. Verifies improvement using the same frozen scenarios
4. Repeats if `max_iterations > 1` (stops on regression)

## Features

### 🎯 Realistic Skill Discovery

Skills Arena tests **real skill discovery** — skills are loaded naturally into the agent's context, exactly how your users experience it. No prompt injection, no artificial setup.

### 📊 Detailed Results with Reasoning

See exactly **why** the agent chose each skill:

```
[Scenario 1]
  Prompt: Find the latest AI news and summarize findings
  Designed for: My Skill
  Selected: Competitor Skill
  Agent's reasoning: I'll help you research AI news. Let me use the
                      competitor skill which handles web research...
```

### 🔴 Steal Detection

Know when competitors win scenarios **designed for your skill**:

```
🔴 STEAL DETECTION
  My Skill: Lost 2 scenario(s) to competitors
    - scenario-abc123
    - scenario-def456
```

### 🎮 Custom Scenarios (Power Users)

Define your own test cases for regression testing, edge cases, or real production prompts:

```python
from skills_arena import Arena, CustomScenario

results = arena.compare(
    skills=["./my-skill.md", "./competitor.md"],
    scenarios=[
        CustomScenario(prompt="Find AI news"),  # Blind test
        CustomScenario(
            prompt="Scrape pricing from stripe.com",
            expected_skill="My Skill",  # Enables steal detection
        ),
    ],
)
```

### 🔀 Mix Custom + Generated Scenarios

```python
from skills_arena import CustomScenario, GenerateScenarios

results = arena.compare(
    skills=["./my-skill.md", "./competitor.md"],
    task="web search",
    scenarios=[
        CustomScenario(prompt="My edge case"),
        GenerateScenarios(count=5),  # Generate 5 more with LLM
    ],
)
```

## Configuration

```python
from skills_arena import Arena, Config

config = Config(
    # Scenario generation
    scenarios=10,                       # Number of test scenarios
    scenario_strategy="per_skill",      # "per_skill" or "balanced"
    temperature=0.7,                    # Generation diversity

    # Agent framework
    agents=["claude-code"],             # Uses Claude Agent SDK

    # Execution
    timeout_seconds=60,                 # Per-scenario timeout
)

arena = Arena(config)
```

### Scenario Strategies

| Strategy | Description |
|----------|-------------|
| `balanced` | Generate scenarios for all skills together (default) |
| `per_skill` | Generate from each skill alone — reveals "steal rates" |

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Required
```

## API Reference

### Arena Methods

| Method | Description |
|--------|-------------|
| `arena.evaluate(skill, task)` | Evaluate a single skill |
| `arena.compare(skills, task)` | Compare multiple skills head-to-head |
| `arena.battle_royale(skills, task)` | Full tournament with ELO rankings |
| `arena.optimize(skill, competitors, task)` | Auto-improve a skill description |

### Result Objects

```python
# ComparisonResult
results.winner              # Name of winning skill
results.selection_rates     # {skill_name: rate}
results.scenario_details    # List of ScenarioDetail
results.steals              # {skill_name: [stolen_scenario_ids]}
results.insights            # List of Insight

# ScenarioDetail
detail.prompt               # The test prompt
detail.expected_skill       # Which skill it was designed for
detail.selected_skill       # Which skill the agent chose
detail.reasoning            # Agent's text before selection
detail.was_stolen           # True if competitor won

# OptimizationResult
result.original_skill       # Skill before optimization
result.optimized_skill      # Best skill found
result.iterations           # List of OptimizationIteration
result.total_improvement    # Delta in selection rate
result.selection_rate_before  # Starting selection rate
result.selection_rate_after   # Final selection rate
result.grade_before         # Grade before (A+ to F)
result.grade_after          # Grade after
```

### Custom Scenarios

```python
from skills_arena import CustomScenario, GenerateScenarios

# Blind test (no expected skill)
CustomScenario(prompt="Find AI news")

# With expected skill (enables steal detection)
CustomScenario(
    prompt="Scrape the pricing table",
    expected_skill="Web Scraper",
    tags=["scraping", "pricing"],
)

# Generate N scenarios with LLM
GenerateScenarios(count=5)
```

## Key Metrics

| Metric | Description | What It Means |
|--------|-------------|---------------|
| **Selection Rate** | % of times your skill is chosen | Your share of the context layer |
| **Steal Rate** | % of your scenarios won by competitors | Opportunities lost to alternatives |
| **Defense Rate** | % of your scenarios you kept | How well you hold your ground |

## Supported Agents

| Agent | Status | Notes |
|-------|--------|-------|
| **Claude Code** | ✅ Supported | Primary agent, uses Claude Agent SDK |
| **Codex CLI** | 🔜 Coming | OpenAI's coding agent |
| **Gemini CLI** | 🔜 Coming | Google's coding agent |
| **Cursor** | 🔜 Planned | IDE-integrated agent |
| **Windsurf** | 🔜 Planned | Codeium's coding agent |

## Supported Skill Formats

- **Claude Code** — `.md` skill files with YAML frontmatter
- **OpenAI** — Function calling schemas (JSON)
- **MCP** — Tool definitions
- **Generic** — Plain text descriptions

## Roadmap

- [x] Filesystem-based skill discovery
- [x] Custom scenarios for power users
- [x] Agent's reasoning capture
- [x] Steal detection
- [x] Auto-optimize skill descriptions
- [ ] Web UI dashboard
- [ ] Historical tracking & trends
- [ ] [skills.sh](https://skills.sh) integration

## Contributing

Contributions welcome! See [ARCHITECTURE.md](./ARCHITECTURE.md) for technical details.

```bash
git clone https://github.com/Eyalbenba/skills-arena.git
cd skills-arena
pip install -e ".[dev]"
pytest
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Skills Arena</strong> — Skills are the new SEO.
</p>
