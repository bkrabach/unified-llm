# unified-llm

Unified LLM client library for Rust — multi-provider support for OpenAI, Anthropic, and Google Gemini.

[![CI](https://github.com/bkrabach/unified-llm/actions/workflows/ci.yaml/badge.svg)](https://github.com/bkrabach/unified-llm/actions)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

`unified-llm` provides a single Rust API for calling large language models across OpenAI, Anthropic (Claude), and Google Gemini. It auto-detects the provider from environment variables, normalizes request/response formats, and supports streaming and tool use through a consistent interface built on Tokio.

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

## Origin

This project was built from the [Unified LLM Client Specification](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md) (NLSpec) by [strongDM](https://github.com/strongdm). The NLSpec defines a language-agnostic specification for building a unified client library across multiple LLM providers.

## Ecosystem

| Project | Description |
|---------|-------------|
| [attractor](https://github.com/bkrabach/attractor) | DOT-based pipeline engine |
| [attractor-server](https://github.com/bkrabach/attractor-server) | HTTP API server |
| [attractor-ui](https://github.com/bkrabach/attractor-ui) | Web frontend |
| [unified-llm](https://github.com/bkrabach/unified-llm) | Multi-provider LLM client |
| [coding-agent-loop](https://github.com/bkrabach/coding-agent-loop) | Agentic tool loop |

## License

MIT
