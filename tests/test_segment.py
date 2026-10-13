from drybench.segment import join_spans, segment


def _check_offsets(text, spans):
    for s in spans:
        assert text[s.start:s.end] == s.text


def test_sentences_and_offsets():
    text = "Great question! The carrier was Ostrell Freight. It used 23 containers. Hope this helps."
    spans = segment(text)
    assert [s.text for s in spans] == [
        "Great question!", "The carrier was Ostrell Freight.", "It used 23 containers.", "Hope this helps."]
    _check_offsets(text, spans)


def test_no_split_on_decimals_or_lowercase_continuation():
    text = "The value is 3.5 tonnes. e.g. this stays together? Yes."
    spans = segment(text)
    assert [s.text for s in spans] == ["The value is 3.5 tonnes. e.g. this stays together?", "Yes."]


def test_lines_and_bullets():
    text = "Summary:\n- Carrier: Ostrell Freight\n- Cargo: 12 crates of tin\n\nThanks!"
    spans = segment(text)
    assert [s.text for s in spans] == ["Summary:", "- Carrier: Ostrell Freight", "- Cargo: 12 crates of tin", "Thanks!"]
    _check_offsets(text, spans)


def test_clause_mode():
    text = "The carrier was X; the cargo was Y. Done."
    assert [s.text for s in segment(text, "clause")] == ["The carrier was X;", "the cargo was Y.", "Done."]


def test_join_spans():
    spans = segment("A one. B two. C three.")
    assert join_spans(spans, [True, False, True]) == "A one. C three."
    assert join_spans(spans) == "A one. B two. C three."


def test_list_enumerators_stay_with_their_item():
    text = "Here you go:\n1. **Carrier:** Ostrell Freight\n2. **Cargo:** 12 crates of tin"
    assert [s.text for s in segment(text)] == [
        "Here you go:", "1. **Carrier:** Ostrell Freight", "2. **Cargo:** 12 crates of tin"]
    inline = "1. The carrier was X. 2. The cargo was Y."
    assert [s.text for s in segment(inline)] == ["1. The carrier was X.", "2. The cargo was Y."]


def test_no_split_after_initials_and_abbreviations():
    """v1.2: 62 development responses were cut after an initial or an abbreviation."""
    cases = {
        "The IPCC was chaired by Rajendra K. Pachauri. He was born in 1940.":
            ["The IPCC was chaired by Rajendra K. Pachauri.", "He was born in 1940."],
        "This is the Pizza Fusion Holdings, Inc. Franchise Agreement.":
            ["This is the Pizza Fusion Holdings, Inc. Franchise Agreement."],
        "It was signed on Nov. 02, 2019. It ends later.": ["It was signed on Nov. 02, 2019.", "It ends later."],
        "Dr. Nick met the U.S. Army at St. Peter's. It is No. 5 on the list.":
            ["Dr. Nick met the U.S. Army at St. Peter's.", "It is No. 5 on the list."],
        "He said no. Then he left.": ["He said no.", "Then he left."],
    }
    for text, want in cases.items():
        spans = segment(text)
        assert [s.text for s in spans] == want, text
        _check_offsets(text, spans)
