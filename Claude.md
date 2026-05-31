


# FleetOps AI - Multi-Agent System

Personal portfolio project demonstrating multi-agent architecture for transportation fleet operations optimization.

## Architecture
- 4 specialized agents running in parallel:
  - Maintenance Forecasting Agent
  - Weather Impact Agent
  - Staffing Coverage Agent
  - Mission Demand Agent
- 1 Orchestration Agent that consumes their outputs and produces recommendations

## Tech Stack
- Python 3.11+
- LangGraph for agent orchestration
- Claude API (Anthropic) for agent reasoning
- Snowflake for data layer
- Streamlit for front end
- Open-Meteo API for weather data (free, no key required)

## Constraints
- Generic transportation fleet use case — NOT tied to any specific company
- Synthetic data only
- Build incrementally — one agent working in isolation before connecting

## My background
Senior data engineer, strong Python/Snowflake/Azure, used LangChain before but new to LangGraph.

## Working style
- Explain LangGraph concepts as we encounter them
- Suggest simplest working version first
- Flag when I'm overcomplicating
- Catch gaps in understanding rather than just writing code

## Commit and code style requirements

- Do NOT add "Co-authored-by: Claude" or any Claude/AI attribution to commit messages
- Do NOT add "🤖 Generated with Claude Code" footers to commits
- Commit messages should be normal, terse, human-style — no AI signatures
- Code should look human-written:
  - No excessive comments explaining obvious things
  - No emoji in code, comments, or commit messages
  - Standard variable names, not overly descriptive ones
  - Don't over-engineer error handling for simple cases
  - Skip docstrings on trivial functions
  - Don't add "as a best practice" type comments