# unified-llm

Unified LLM client library for Rust — multi-provider support for OpenAI, Anthropic, and Google Gemini.

[![CI](https://github.com/bkrabach/unified-llm/actions/workflows/ci.yaml/badge.svg)](https://github.com/bkrabach/unified-llm/actions)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Multi-provider** — OpenAI, Anthropic (Claude), and Google Gemini through a single API
- **Auto-detection** — Picks the right provider from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)
- **Streaming** — SSE-based streaming for all providers
- **Tool use** — Unified tool calling across providers
- **Async** — Built on Tokio for async I/O

## Quick Start

```bash
git clone https://github.com/bkrabach/unified-llm.git
cd unified-llm
cargo build
```

```rust
use unified_llm::{Client, GenerateParams, generate};

let client = Client::from_env()?;
let params = GenerateParams::new("gpt-4o", "Hello, world!");
let result = generate(params).await?;
println!("{}", result.text);
```

## Environment Variables

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `GOOGLE_API_KEY` | Google Gemini |

## Related

- [attractor](https://github.com/bkrabach/attractor) — Pipeline engine (uses unified-llm)
- [coding-agent-loop](https://github.com/bkrabach/coding-agent-loop) — Agentic tool loop (uses unified-llm)
