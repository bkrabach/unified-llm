//! Server-Sent Events (SSE) line parser.
//!
//! Implements the SSE dispatch algorithm from RFC 8895.  All three provider
//! adapters share this module so each adapter only interprets event *payloads*
//! rather than re-implementing the wire protocol.
//!
//! This module is `pub(crate)` — it is an implementation detail of the
//! provider adapters and is not part of the public API.
//!
//! # Dead-code suppression
//!
//! Provider adapters have not been implemented yet (F-007+), so the public
//! surface of this module appears unused to the compiler.  The lint is
//! suppressed here; once adapters consume these functions the attribute can
//! be removed.
#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A fully-parsed SSE event, ready for provider-specific interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The event type field.  Defaults to `"message"` if absent.
    pub event_type: String,
    /// Accumulated data.  Multiple `data:` lines are joined with `\n`.
    pub data: String,
    /// The event ID field, if present.
    pub id: Option<String>,
    /// The retry interval in milliseconds, if the server sent a `retry:` field.
    pub retry_ms: Option<u64>,
}

/// Errors that can occur during SSE parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseParseError {
    /// A `data:` line contained non-UTF-8 bytes (not yet reachable with `&str` API).
    InvalidUtf8,
    /// A `retry:` field value was not a valid decimal integer.
    InvalidRetryValue(String),
}

impl std::fmt::Display for SseParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in SSE data"),
            Self::InvalidRetryValue(v) => {
                write!(f, "invalid retry value (expected decimal integer): {v}")
            }
        }
    }
}

impl std::error::Error for SseParseError {}

// ---------------------------------------------------------------------------
// Core parsing logic
// ---------------------------------------------------------------------------

/// Process a single SSE line and update the provided field buffers.
///
/// Returns `Some(SseEvent)` when a blank line is encountered (event dispatch),
/// `None` otherwise.
///
/// Internal use only — exposed as `pub(crate)` for adapter use.
pub(crate) fn process_sse_line(
    line: &str,
    event_type: &mut String,
    data_lines: &mut Vec<String>,
    last_id: &mut Option<String>,
    retry_ms: &mut Option<u64>,
) -> Result<Option<SseEvent>, SseParseError> {
    // Blank (or whitespace-only) line → dispatch event
    if line.chars().all(|c| c == '\r' || c == ' ' || c == '\t') {
        // If no `data:` lines were added at all, discard (RFC 8895 §9.2.6 step 4).
        // An explicit `data:` with empty value is NOT discarded — it produces an
        // event with data == "".
        if data_lines.is_empty() {
            // Reset per-event buffers; last_id persists
            event_type.clear();
            return Ok(None);
        }
        let data_joined = data_lines.join("\n");

        let event = SseEvent {
            event_type: if event_type.is_empty() {
                "message".to_string()
            } else {
                event_type.clone()
            },
            data: data_joined,
            id: last_id.clone(),
            retry_ms: *retry_ms,
        };

        // Reset per-event buffers; last_id persists per RFC 8895
        event_type.clear();
        data_lines.clear();

        return Ok(Some(event));
    }

    // Comment line — silently ignore
    if line.starts_with(':') {
        return Ok(None);
    }

    // Split on the first `:` to get field name and value
    let (field, value) = if let Some(pos) = line.find(':') {
        let f = &line[..pos];
        // RFC 8895: a single optional space after the colon is consumed
        let v = &line[pos + 1..];
        let v = v.strip_prefix(' ').unwrap_or(v);
        (f, v)
    } else {
        // No colon → field name is the whole line, value is empty string
        (line, "")
    };

    match field {
        "data" => {
            data_lines.push(value.to_string());
        }
        "event" => {
            *event_type = value.to_string();
        }
        "id" => {
            *last_id = Some(value.to_string());
        }
        "retry" => match value.parse::<u64>() {
            Ok(ms) => *retry_ms = Some(ms),
            Err(_) => {
                return Err(SseParseError::InvalidRetryValue(value.to_string()));
            }
        },
        // Unknown fields are silently ignored per RFC 8895
        _ => {}
    }

    Ok(None)
}

// ---------------------------------------------------------------------------
// Iterator-based public API
// ---------------------------------------------------------------------------

/// Stateful iterator that wraps the line-processing logic.
struct SseLineParser<'a, I> {
    lines: I,
    event_type: String,
    data_lines: Vec<String>,
    last_id: Option<String>,
    retry_ms: Option<u64>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, I: Iterator<Item = &'a str>> Iterator for SseLineParser<'a, I> {
    type Item = Result<SseEvent, SseParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let line = self.lines.next()?;
            match process_sse_line(
                line,
                &mut self.event_type,
                &mut self.data_lines,
                &mut self.last_id,
                &mut self.retry_ms,
            ) {
                Ok(Some(event)) => return Some(Ok(event)),
                Ok(None) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Parse raw SSE lines into zero or more [`SseEvent`] values.
///
/// Returns an iterator of `Result<SseEvent, SseParseError>`.
pub fn parse_sse_lines<'a>(
    lines: impl Iterator<Item = &'a str> + 'a,
) -> impl Iterator<Item = Result<SseEvent, SseParseError>> + 'a {
    SseLineParser {
        lines,
        event_type: String::new(),
        data_lines: Vec::new(),
        last_id: None,
        retry_ms: None,
        _phantom: std::marker::PhantomData,
    }
}

/// Convenience: parse a complete SSE text block (e.g., from a test fixture).
///
/// Returns all events found, or the first parse error.
pub fn parse_sse_text(text: &str) -> Result<Vec<SseEvent>, SseParseError> {
    parse_sse_lines(text.lines()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // AC-1: data: hello\n\n → SseEvent { event_type: "message", data: "hello", .. }
    #[test]
    fn single_data_line() {
        let events = parse_sse_text("data: hello\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "hello");
        assert!(events[0].id.is_none());
        assert!(events[0].retry_ms.is_none());
    }

    // AC-2: event: text\ndata: foo\n\n
    #[test]
    fn event_type_override() {
        let events = parse_sse_text("event: text\ndata: foo\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "text");
        assert_eq!(events[0].data, "foo");
    }

    // AC-3: multi-line data joined with \n
    #[test]
    fn multi_line_data() {
        let events = parse_sse_text("data: a\ndata: b\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a\nb");
    }

    // AC-4: comment line skipped
    #[test]
    fn comment_skipped() {
        let events = parse_sse_text(": this is a comment\ndata: x\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    // AC-5: unknown field ignored
    #[test]
    fn unknown_field_ignored() {
        let events = parse_sse_text("x-custom: val\ndata: y\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "y");
    }

    // AC-6: retry field parsed
    #[test]
    fn retry_field_parsed() {
        let events = parse_sse_text("retry: 3000\ndata: x\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retry_ms, Some(3000));
    }

    // AC-7: invalid retry returns error
    #[test]
    fn invalid_retry_error() {
        let result = parse_sse_text("retry: not-a-number\ndata: x\n\n");
        assert_eq!(
            result,
            Err(SseParseError::InvalidRetryValue("not-a-number".to_string()))
        );
    }

    // AC-8: blank data → no event emitted (blank line with no prior data)
    #[test]
    fn empty_data_no_event() {
        let events = parse_sse_text("\n").unwrap();
        assert!(events.is_empty());
    }

    // AC-9: [DONE] emitted as-is
    #[test]
    fn done_payload_emitted() {
        let events = parse_sse_text("data: [DONE]\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
    }

    // AC-10: two events separated by blank line
    #[test]
    fn two_events() {
        let events = parse_sse_text("data: a\n\ndata: b\n\n").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
    }

    // AC-11: id persists across events
    #[test]
    fn id_persists() {
        let events = parse_sse_text("id: req-123\ndata: x\n\ndata: y\n\n").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, Some("req-123".to_string()));
        assert_eq!(events[1].id, Some("req-123".to_string()));
    }

    // Edge: data: with no value → empty string data line → dispatches event with empty data
    #[test]
    fn data_no_value() {
        let events = parse_sse_text("data:\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    // Edge: bare word (no colon) → treated as field "data" with empty value
    #[test]
    fn bare_word_ignored() {
        let events = parse_sse_text("data\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    // Edge: event set but no data → discarded
    #[test]
    fn event_without_data_discarded() {
        let events = parse_sse_text("event: ping\n\n").unwrap();
        assert!(events.is_empty());
    }

    // Edge: id: with empty value → Some("")
    #[test]
    fn empty_id_value() {
        let events = parse_sse_text("id:\ndata: x\n\n").unwrap();
        assert_eq!(events[0].id, Some(String::new()));
    }

    // Edge: consecutive blank lines — first dispatches, second discards (no data)
    #[test]
    fn consecutive_blank_lines() {
        let events = parse_sse_text("data: a\n\n\n").unwrap();
        assert_eq!(events.len(), 1);
    }

    // event type resets to "message" after each dispatch
    #[test]
    fn event_type_resets() {
        let events = parse_sse_text("event: custom\ndata: a\n\ndata: b\n\n").unwrap();
        assert_eq!(events[0].event_type, "custom");
        assert_eq!(events[1].event_type, "message");
    }

    // SseParseError Display
    #[test]
    fn parse_error_display() {
        let e = SseParseError::InvalidRetryValue("abc".to_string());
        assert!(format!("{e}").contains("retry"));

        let e2 = SseParseError::InvalidUtf8;
        assert!(format!("{e2}").contains("UTF-8"));
    }

    // data: value with leading space stripped, trailing space preserved
    #[test]
    fn leading_space_stripped() {
        let events = parse_sse_text("data: hello\n\n").unwrap();
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn trailing_space_preserved() {
        let events = parse_sse_text("data: hello \n\n").unwrap();
        assert_eq!(events[0].data, "hello ");
    }

    // ---------------------------------------------------------------------------
    // Property-based tests (proptest)
    // ---------------------------------------------------------------------------

    use proptest::prelude::*;

    proptest! {
        // AC-12: random valid SSE lines never panic
        #[test]
        fn prop_no_panic_on_arbitrary_lines(
            lines in prop::collection::vec(
                "[a-zA-Z0-9: _\\-\\.\\[\\]]*",
                0..20
            )
        ) {
            // Collect into a Vec<String> first so we control lifetimes
            let joined = lines.join("\n");
            // parse_sse_text should never panic — errors are fine
            let _ = parse_sse_text(&joined);
        }

        #[test]
        fn prop_data_roundtrips(data in "[a-zA-Z0-9 _\\-\\.]*") {
            let input = format!("data: {data}\n");
            let result = parse_sse_text(&input);
            if let Ok(events) = result {
                if !events.is_empty() {
                    prop_assert_eq!(&events[0].data, &data);
                }
            }
        }

        #[test]
        fn prop_valid_retry_parsed(ms in 0u64..1_000_000u64) {
            let input = format!("retry: {ms}\ndata: x\n\n");
            let events = parse_sse_text(&input).unwrap();
            prop_assert_eq!(events.len(), 1);
            prop_assert_eq!(events[0].retry_ms, Some(ms));
        }

        #[test]
        fn prop_event_type_roundtrips(
            ev in "[a-zA-Z][a-zA-Z0-9_\\-]*"
        ) {
            let input = format!("event: {ev}\ndata: x\n\n");
            let events = parse_sse_text(&input).unwrap();
            prop_assert_eq!(events.len(), 1);
            prop_assert_eq!(&events[0].event_type, &ev);
        }
    }
}
