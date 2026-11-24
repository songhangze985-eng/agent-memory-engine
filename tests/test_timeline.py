from agentmem_core.timeline import TimelineBook, record_timeline_buffered


def test_merge_extends_t_end_and_overwrites_payload():
    layer = []
    record_timeline_buffered(layer, {"origin": "A"}, 100, delta_t=10)
    record_timeline_buffered(layer, {"origin": "B"}, 110, delta_t=10)
    assert len(layer) == 1
    assert layer[0]["t_start"] == 100
    assert layer[0]["t_end"] == 110
    assert layer[0]["origin"] == "B"


def test_gap_above_delta_appends_a_node():
    layer = []
    record_timeline_buffered(layer, {"origin": "A"}, 100, delta_t=10)
    record_timeline_buffered(layer, {"origin": "B"}, 110.1, delta_t=10)
    assert len(layer) == 2
    assert layer[1]["origin"] == "B"
    assert layer[0]["origin"] == "A"


def test_moving_t_end_keeps_a_busy_session_in_one_node():
    book = TimelineBook(delta_t=10, enabled=True)
    book.record(1, {"origin": "A"}, 0)
    book.record(1, {"origin": "B"}, 5)
    book.record(1, {"origin": "C"}, 10)
    assert len(book.layer1) == 1
    assert book.layer1[0]["t_end"] == 10
    assert book.layer1[0]["origin"] == "C"
    book.record(1, {"origin": "D"}, 21)
    assert len(book.layer1) == 2


def test_layers_are_independent_and_buffer_can_be_disabled():
    book = TimelineBook(delta_t=10, enabled=True)
    book.record(1, {"origin": "A"}, 1)
    book.record(2, {"in_cnt": 1, "out_cnt": 4}, 1)
    book.record(2, {"in_cnt": 2, "out_cnt": 3}, 3)
    assert len(book.layer1) == 1
    assert len(book.layer2) == 1
    assert book.layer2[0]["in_cnt"] == 2
    assert book.layer2[0]["out_cnt"] == 3

    plain = TimelineBook(delta_t=10, enabled=False)
    plain.record(1, {"origin": "A"}, 1)
    plain.record(1, {"origin": "B"}, 1)
    assert len(plain.layer1) == 2
    assert plain.reduction() == 0
