from Demo import short_long_router


def test_keyword_long_hints_force_long_mode() -> None:
    decision = short_long_router.route_query("Can you explain how to fix this issue step by step?")
    assert decision.mode == "LONG"
    assert decision.confidence == 0.95


def test_keyword_short_hints_force_short_mode() -> None:
    decision = short_long_router.route_query("Can you answer this just the answer quickly?")
    assert decision.mode == "SHORT"
    assert decision.confidence == 0.9


def test_empty_query_defaults_short() -> None:
    decision = short_long_router.route_query("")
    assert decision.mode == "SHORT"
    assert decision.reason == "empty_input"
