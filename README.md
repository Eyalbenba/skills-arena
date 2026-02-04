# Skills Arena

> **The SEO platform for AI agent skills** - Benchmark, optimize, and compete your skill descriptions.

## The Problem

You've built an amazing skill for your AI agent. But when a user asks a question, **will the agent actually pick your skill?**

Just like websites compete for search engine rankings, skills compete for agent selection. Skills Arena helps you:

- **Measure** how often agents choose your skill
- **Compare** your skill against competitors
- **Optimize** descriptions for better selection rates
- **Test** across Claude, GPT, Gemini, and more

## Installation

```bash
pip install skills-arena
```

## Quick Start

### Evaluate a Single Skill

```python
from skills_arena import Arena

# One line to evaluate
results = Arena().evaluate("./my-skill.md", task="web search")

print(results.score)        # 78.5
print(results.grade)        # "B+"
print(results.selection_rate)  # 0.78
```

### Compare Two Skills

```python
from skills_arena import Arena

arena = Arena()
results = arena.compare(
    skills=["./my-skill.md", "./competitor.md"],
    task="web search and content extraction",
)

print(results.winner)           # "competitor"
print(results.selection_rates)  # {"my-skill": 0.42, "competitor": 0.58}
print(results.insights[0])      # "Your skill lacks specific examples..."
```

### Battle Royale (Multiple Skills)

```python
from skills_arena import Arena, Config

config = Config(
    scenarios=100,
    agents=["claude-code"],  # Test against Claude Code agent
)

arena = Arena(config)
results = arena.battle_royale(
    skills=["./skill-a.md", "./skill-b.md", "./skill-c.md"],
    task="data analysis",
)

print(results.leaderboard)
# 1. skill-b (ELO: 1523)
# 2. skill-a (ELO: 1489)
# 3. skill-c (ELO: 1388)
```

### Using Config File

```python
from skills_arena import Arena

arena = Arena.from_config("./arena.yaml")
results = arena.run()
```

```yaml
# arena.yaml
task: "web search and information retrieval"

skills:
  - ./skills/my-search.md
  - ./skills/competitor.md

evaluation:
  scenarios: 50
  agents: [claude-code]  # or: [claude-code, raw-openai]
  mode: compare
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Your Skill     │────▶│    Scenario     │────▶│  Arena Runner   │
│  Description    │     │    Generator    │     │  (Multi-Agent)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌───────────────┐          │
                              │ Selection Rate│◀─────────┘
                              │    & Score    │
                              └───────────────┘
```

1. **Parse** your skill description (Claude Code, OpenAI, MCP formats)
2. **Generate** diverse test scenarios from task description
3. **Run** scenarios through agents with competing skills
4. **Score** based on selection rate and accuracy
5. **Report** insights and optimization suggestions

## Key Metrics

| Metric | Description | SEO Analogy |
|--------|-------------|-------------|
| **Selection Rate** | % of times your skill is chosen | Click-through rate |
| **Preference Score** | ELO ranking vs competitors | Search position |
| **Invocation Accuracy** | Correct usage when selected | Conversion rate |
| **Description Efficiency** | Quality per token | Page speed |

## Supported Formats

- **Claude Code** - `.md` skill files
- **OpenAI** - Function calling schemas (JSON)
- **MCP** - Tool definitions
- **Generic** - Plain text descriptions

## Supported Agent Frameworks

We test against **real agent frameworks**, not just raw LLM APIs:

| Framework | Status | Notes |
|-----------|--------|-------|
| **Claude Code** | ✅ Primary | Uses Claude Code SDK (Python) |
| **Codex** | ⚠️ Phase 2 | TypeScript-only, requires bridge |
| **Raw Claude API** | ✅ Fallback | Direct tool_use (no agent logic) |
| **Raw OpenAI API** | ✅ Fallback | Direct function_calling (no agent logic) |

> **Why agent frameworks matter**: Claude Code and Codex have their own skill selection
> logic, system prompts, and behaviors. Testing against raw APIs misses these real-world factors.

## Configuration

```python
from skills_arena import Config

config = Config(
    # Scenario generation
    scenarios=50,                      # Number of test scenarios
    temperature=0.7,                   # Generation diversity
    include_adversarial=True,          # Edge cases

    # Agent frameworks to test against
    # Options: "claude-code", "codex", "raw-claude", "raw-openai"
    agents=["claude-code"],            # Primary: Claude Code SDK

    # Execution
    parallel_requests=10,              # Concurrency

    # Output
    verbose=True,                      # Progress feedback
)
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

## Example Output

```
Skills Arena - Comparison Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Task: web search and content extraction
Scenarios: 100
Agents: claude-sonnet, gpt-4o

Results:
┌──────────────────┬─────────────┬────────────┐
│ Skill            │ Selection % │ Wins       │
├──────────────────┼─────────────┼────────────┤
│ tavily-search    │ 58%         │ 58         │
│ my-search-skill  │ 42%         │ 42         │
└──────────────────┴─────────────┴────────────┘

Winner: tavily-search (+16 advantage)

Insights:
• Your skill description is 40% longer but less specific
• Competitor has clearer "when to use" examples
• Consider adding: API response format documentation
```

## Roadmap

- [x] Architecture design
- [x] Phase 1: Core SDK (`evaluate`, `compare`)
- [ ] Phase 2: Multi-framework support
- [ ] Phase 3: Battle royale & ELO rankings
- [ ] Phase 4: AI insights engine
- [ ] Phase 5: Web UI & ecosystem
- [ ] Phase 6: [skills.sh](https://skills.sh) integration - Browse public skills registry and compare head-to-head

## Contributing

This project is in early development. Contributions welcome!

See [ARCHITECTURE.md](./ARCHITECTURE.md) for technical details.

## License

MIT

---

*Skills Arena - Because your skill deserves to be chosen.*
