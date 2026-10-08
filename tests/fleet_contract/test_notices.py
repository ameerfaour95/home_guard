"""Owner-notice kinds are data: the box and the owner's app branch on `kind`, never on the message text."""
import pytest

from home_guard_project.fleet_contract.notices import NOTICE_KINDS, check_kind, notice_message


def test_the_kinds_and_their_sentences():
    assert NOTICE_KINDS == ("recording", "chat")
    assert notice_message("chat", [], "10:05") == "Home Guard support viewed your chat with the assistant (10:05)"
    assert notice_message("recording", [], "10:05–10:20") == "Home Guard support viewed recordings (10:05–10:20)"
    assert notice_message("recording", ["Front door", "Gate", "Pool"], "10:05") == \
        "Home Guard support viewed recordings from Front door, Gate and Pool (10:05)"


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown notice kind"):
        check_kind("live_view")
    with pytest.raises(ValueError):
        notice_message("thumbnail", [], "10:05")
