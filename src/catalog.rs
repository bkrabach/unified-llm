//! Static model catalog for `unified-llm`.
//!
//! Provides lookup functions for known LLM models across OpenAI, Anthropic,
//! and Google Gemini. The catalog is a compile-time constant initialized once
//! via [`std::sync::OnceLock`].
//!
//! # Design
//! - Catalog data is stored in a lazily-initialized `Vec<ModelInfo>`.
//! - All public functions return `&'static ModelInfo` references so callers
//!   pay no allocation cost.
//! - Alias lookup uses a linear scan (catalog size < 50 entries).

use std::sync::OnceLock;

use crate::types::ModelInfo;

// ---------------------------------------------------------------------------
// Static catalog
// ---------------------------------------------------------------------------

static CATALOG: OnceLock<Vec<ModelInfo>> = OnceLock::new();

fn catalog() -> &'static [ModelInfo] {
    CATALOG.get_or_init(|| {
        vec![
            // ── OpenAI ────────────────────────────────────────────────────
            ModelInfo {
                id: "gpt-4o".to_string(),
                provider: "openai".to_string(),
                display_name: "GPT-4o".to_string(),
                context_window: 128_000,
                max_output: Some(16_384),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: false,
                input_cost_per_million: Some(2.5),
                output_cost_per_million: Some(10.0),
                aliases: vec!["gpt-4o-latest".to_string()],
            },
            ModelInfo {
                id: "gpt-4o-mini".to_string(),
                provider: "openai".to_string(),
                display_name: "GPT-4o Mini".to_string(),
                context_window: 128_000,
                max_output: Some(16_384),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: false,
                input_cost_per_million: Some(0.15),
                output_cost_per_million: Some(0.60),
                aliases: vec![],
            },
            ModelInfo {
                id: "o3".to_string(),
                provider: "openai".to_string(),
                display_name: "o3".to_string(),
                context_window: 200_000,
                max_output: Some(100_000),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: true,
                input_cost_per_million: Some(10.0),
                output_cost_per_million: Some(40.0),
                aliases: vec!["o3-latest".to_string()],
            },
            ModelInfo {
                id: "o4-mini".to_string(),
                provider: "openai".to_string(),
                display_name: "o4 Mini".to_string(),
                context_window: 200_000,
                max_output: Some(100_000),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: true,
                input_cost_per_million: Some(1.1),
                output_cost_per_million: Some(4.4),
                aliases: vec![],
            },
            // ── Anthropic ─────────────────────────────────────────────────
            ModelInfo {
                id: "claude-opus-4-5".to_string(),
                provider: "anthropic".to_string(),
                display_name: "Claude Opus 4.5".to_string(),
                context_window: 200_000,
                max_output: Some(32_000),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: true,
                input_cost_per_million: Some(15.0),
                output_cost_per_million: Some(75.0),
                aliases: vec!["claude-opus-latest".to_string()],
            },
            ModelInfo {
                id: "claude-sonnet-4-5".to_string(),
                provider: "anthropic".to_string(),
                display_name: "Claude Sonnet 4.5".to_string(),
                context_window: 200_000,
                max_output: Some(64_000),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: true,
                input_cost_per_million: Some(3.0),
                output_cost_per_million: Some(15.0),
                aliases: vec!["claude-sonnet-latest".to_string()],
            },
            ModelInfo {
                id: "claude-haiku-3-5".to_string(),
                provider: "anthropic".to_string(),
                display_name: "Claude Haiku 3.5".to_string(),
                context_window: 200_000,
                max_output: Some(8_192),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: false,
                input_cost_per_million: Some(0.8),
                output_cost_per_million: Some(4.0),
                aliases: vec!["claude-haiku-latest".to_string()],
            },
            // ── Gemini ────────────────────────────────────────────────────
            ModelInfo {
                id: "gemini-2.5-pro".to_string(),
                provider: "gemini".to_string(),
                display_name: "Gemini 2.5 Pro".to_string(),
                context_window: 1_048_576,
                max_output: Some(65_536),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: true,
                input_cost_per_million: Some(1.25),
                output_cost_per_million: Some(10.0),
                aliases: vec!["gemini-pro-latest".to_string()],
            },
            ModelInfo {
                id: "gemini-2.5-flash".to_string(),
                provider: "gemini".to_string(),
                display_name: "Gemini 2.5 Flash".to_string(),
                context_window: 1_048_576,
                max_output: Some(65_536),
                supports_tools: true,
                supports_vision: true,
                supports_reasoning: true,
                input_cost_per_million: Some(0.075),
                output_cost_per_million: Some(0.30),
                aliases: vec!["gemini-flash-latest".to_string()],
            },
        ]
    })
}

// ---------------------------------------------------------------------------
// Public lookup functions
// ---------------------------------------------------------------------------

/// Returns a model's metadata by exact ID or registered alias.
///
/// Searches canonical IDs first (priority over aliases). Returns `None` if the
/// ID is not recognized.
///
/// ```rust
/// # use unified_llm::catalog::get_model_info;
/// let info = get_model_info("gpt-4o").unwrap();
/// assert_eq!(info.provider, "openai");
///
/// // Aliases also resolve to the canonical entry.
/// let same = get_model_info("gpt-4o-latest").unwrap();
/// assert_eq!(same.id, "gpt-4o");
/// ```
pub fn get_model_info(model_id: &str) -> Option<&'static ModelInfo> {
    let cat = catalog();
    // Canonical IDs take priority over aliases.
    if let Some(m) = cat.iter().find(|m| m.id == model_id) {
        return Some(m);
    }
    cat.iter().find(|m| m.aliases.iter().any(|a| a == model_id))
}

/// Lists all models. If `provider` is `Some`, filters to that provider only.
///
/// Returns a vec of static references **stable-sorted**: provider
/// alphabetically, then model id alphabetically within each provider.
///
/// Provider matching is **case-sensitive**: `"OpenAI"` will not match `"openai"`.
pub fn list_models(provider: Option<&str>) -> Vec<&'static ModelInfo> {
    let mut result: Vec<&'static ModelInfo> = catalog()
        .iter()
        .filter(|m| provider.is_none_or(|p| m.provider == p))
        .collect();
    result.sort_by(|a, b| a.provider.cmp(&b.provider).then_with(|| a.id.cmp(&b.id)));
    result
}

/// Returns the "latest" recommended model for a provider, optionally filtered
/// by a capability tag (`"vision"`, `"reasoning"`, `"tools"`).
///
/// Selection heuristic: among all matching models, returns the one with the
/// largest `context_window`.
///
/// Returns `None` if:
/// - No models are registered for `provider`.
/// - A `capability` was specified but no model for `provider` supports it.
/// - `capability` is an unrecognized string (only `"vision"`, `"reasoning"`,
///   and `"tools"` are recognized).
pub fn get_latest_model(provider: &str, capability: Option<&str>) -> Option<&'static ModelInfo> {
    catalog()
        .iter()
        .filter(|m| m.provider == provider)
        .filter(|m| match capability {
            None => true,
            Some("vision") => m.supports_vision,
            Some("tools") => m.supports_tools,
            Some("reasoning") => m.supports_reasoning,
            Some(_) => false, // unrecognized capability → no match
        })
        .max_by_key(|m| m.context_window)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // AC-1: get_model_info("gpt-4o") returns Some with provider == "openai"
    #[test]
    fn get_model_info_canonical_id() {
        let info = get_model_info("gpt-4o").unwrap();
        assert_eq!(info.provider, "openai");
        assert_eq!(info.id, "gpt-4o");
    }

    // AC-2: alias "claude-sonnet-latest" resolves to claude-sonnet-4-5
    #[test]
    fn get_model_info_alias() {
        let info = get_model_info("claude-sonnet-latest").unwrap();
        assert_eq!(info.id, "claude-sonnet-4-5");
        assert_eq!(info.provider, "anthropic");
    }

    // AC-3: unknown model returns None
    #[test]
    fn get_model_info_unknown_returns_none() {
        assert!(get_model_info("does-not-exist").is_none());
    }

    // AC-4: list_models(None) has at least 9 entries
    #[test]
    fn list_models_all_at_least_nine() {
        assert!(list_models(None).len() >= 9);
    }

    // AC-5: list_models(Some("gemini")) only returns gemini entries
    #[test]
    fn list_models_gemini_only() {
        let models = list_models(Some("gemini"));
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.provider == "gemini"));
    }

    // AC-6: list_models(Some("does-not-exist")) returns empty vec
    #[test]
    fn list_models_nonexistent_provider_empty() {
        assert!(list_models(Some("does-not-exist")).is_empty());
    }

    // AC-7: get_latest_model("openai", Some("reasoning")) returns reasoning model
    #[test]
    fn get_latest_model_reasoning() {
        let m = get_latest_model("openai", Some("reasoning")).unwrap();
        assert!(m.supports_reasoning);
        assert_eq!(m.provider, "openai");
    }

    // AC-8: get_latest_model("nope", None) returns None
    #[test]
    fn get_latest_model_unknown_provider_none() {
        assert!(get_latest_model("nope", None).is_none());
    }

    // AC-9: all catalog entries have non-empty id, provider, display_name
    #[test]
    fn catalog_entries_non_empty_fields() {
        for m in list_models(None) {
            assert!(!m.id.is_empty(), "empty id in catalog");
            assert!(!m.provider.is_empty(), "empty provider in catalog");
            assert!(!m.display_name.is_empty(), "empty display_name in catalog");
        }
    }

    // Edge: canonical ID wins over alias when the same string matches both
    #[test]
    fn canonical_id_wins_over_alias() {
        // "gpt-4o" is a canonical ID → should return gpt-4o entry directly
        let info = get_model_info("gpt-4o").unwrap();
        assert_eq!(info.id, "gpt-4o");
    }

    // Edge: unrecognized capability → None
    #[test]
    fn unknown_capability_returns_none() {
        assert!(get_latest_model("openai", Some("audio")).is_none());
    }

    // Edge: list_models case-sensitive ("OpenAI" ≠ "openai")
    #[test]
    fn list_models_case_sensitive() {
        assert!(list_models(Some("OpenAI")).is_empty());
    }

    // Edge: entry with empty aliases still found by canonical id
    #[test]
    fn model_with_no_aliases_found_by_id() {
        let info = get_model_info("gpt-4o-mini").unwrap();
        assert_eq!(info.id, "gpt-4o-mini");
        assert!(info.aliases.is_empty());
    }

    // list_models is sorted: provider then id
    #[test]
    fn list_models_sorted_order() {
        let models = list_models(None);
        for window in models.windows(2) {
            let cmp = window[0]
                .provider
                .cmp(&window[1].provider)
                .then_with(|| window[0].id.cmp(&window[1].id));
            assert!(
                cmp != std::cmp::Ordering::Greater,
                "list_models out of order: {} {} > {} {}",
                window[0].provider,
                window[0].id,
                window[1].provider,
                window[1].id
            );
        }
    }

    // get_latest_model vision
    #[test]
    fn get_latest_model_vision_openai() {
        let m = get_latest_model("openai", Some("vision")).unwrap();
        assert!(m.supports_vision);
        assert_eq!(m.provider, "openai");
    }

    // get_latest_model anthropic reasoning
    #[test]
    fn get_latest_model_anthropic_reasoning() {
        let m = get_latest_model("anthropic", Some("reasoning")).unwrap();
        assert!(m.supports_reasoning);
        assert_eq!(m.provider, "anthropic");
    }

    // get_latest_model anthropic None → highest context window
    #[test]
    fn get_latest_model_anthropic_none_largest_context() {
        let m = get_latest_model("anthropic", None).unwrap();
        // All anthropic models have 200_000 context window; pick any
        assert_eq!(m.context_window, 200_000);
    }

    // "gpt-4o-latest" alias resolves to gpt-4o
    #[test]
    fn gpt4o_latest_alias_resolves() {
        let m = get_model_info("gpt-4o-latest").unwrap();
        assert_eq!(m.id, "gpt-4o");
    }
}
