from sim.screen import COLS, ROWS, Screen, wrap


def test_question_then_streamed_answer():
    s = Screen()
    s.handle({"type": "transcript", "text": "what plant"})
    s.handle({"type": "delta", "text": "Monstera. "})
    s.handle({"type": "delta", "text": "Water weekly."})
    assert s.lines() == ["> what plant", "Monstera. Water", "weekly."]


def test_status_shows_while_working_and_gives_way_to_the_answer():
    s = Screen()
    s.handle({"type": "transcript", "text": "where did I park"})
    assert s.lines() == ["> where did I park", "Thinking..."]
    s.handle({"type": "status", "text": "Checking notes..."})
    assert s.lines() == ["> where did I park", "Checking notes..."]
    s.handle({"type": "delta", "text": "Level 3."})
    assert s.lines() == ["> where did I park", "Level 3."]


def test_error_replaces_screen():
    s = Screen()
    s.handle({"type": "transcript", "text": "q"})
    s.handle({"type": "delta", "text": "partial"})
    s.handle({"type": "error", "message": "Busy"})
    assert s.lines() == ["ERROR: Busy"]


def test_local_status_starts_fresh():
    s = Screen()
    s.handle({"type": "transcript", "text": "old question"})
    s.handle({"type": "delta", "text": "old answer"})
    s.show("Listening...")
    assert s.lines() == ["Listening..."]


def test_done_returns_timings():
    assert Screen().handle({"type": "done", "timings": {"total": 5}}) == {"total": 5}


def test_lines_fit_the_oled():
    s = Screen()
    s.handle({"type": "delta", "text": "word " * 200})
    lines = s.lines()
    assert len(lines) == ROWS  # scrolled to the newest lines
    assert all(len(line) <= COLS for line in lines)


def test_wrap_keeps_blank_lines():
    assert wrap("a\n\nb") == ["a", "", "b"]
