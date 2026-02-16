from Demo import text_utils


def test_validate_cloud_filler_output_rejects_bad_patterns_and_digits() -> None:
    assert text_utils.validate_cloud_filler_output("Can I answer that?") is None
    assert text_utils.validate_cloud_filler_output("This has 3 steps to know") is None
    assert text_utils.validate_cloud_filler_output("John Smith, I know someone") is None
    assert text_utils.validate_cloud_filler_output("Let us discuss") == "Let us discuss."


def test_fallback_cloud_filler_is_one_of_known_phrases() -> None:
    phrase = text_utils.fallback_cloud_filler("any input")
    assert phrase in {
        "One moment.",
        "Just a sec.",
        "Checking that now.",
        "Working on it.",
        "Let me check.",
    }


def test_should_emit_long_filler_only_after_delay() -> None:
    assert not text_utils.should_emit_long_filler("SHORT", 1.0, 750)
    assert not text_utils.should_emit_long_filler("LONG", 0.5, 750)
    assert text_utils.should_emit_long_filler("LONG", 0.8, 750)


def test_update_memory_contains_recent_turns_summary_and_facts() -> None:
    history = [
        ("my name is Alice", "Nice to meet you, Alice."),
        ("i live in Seoul", "Great place to be."),
        ("I prefer tea", "Tea is a solid choice."),
        ("explain caching", "Caching stores results for reuse."),
    ]
    state = text_utils.update_memory(history, n_recent_turns=2, max_summary_turns=10, summary_word_budget=30)

    assert state["rolling_summary"]
    assert len(state["recent_raw_turns"]) == 2
    assert state["recent_raw_turns"][-1] == ("explain caching", "Caching stores results for reuse.")
    assert state["pinned_facts"].get("name") == "Alice"
    assert state["pinned_facts"].get("location") == "Seoul"
    assert state["pinned_facts"].get("preference") == "tea"
