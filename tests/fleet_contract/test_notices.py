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


def test_the_push_command_is_the_one_agreed_with_the_box():
    import base64
    import json
    from pathlib import Path

    from home_guard_project.fleet_contract import notices as n

    body = {"schema_version": 1, "id": 42, "kind": "recording", "cameras": ["כניסה"], "message": "x"}
    b64 = n.notice_b64(body)
    assert json.loads(base64.b64decode(b64).decode("utf-8")) == body
    assert n.notice_command(body) == (
        r"cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box notices add --b64 " + b64 + " --json")
    assert n.notice_ssh_argv("ameer", "desktop-43dp1ti", body, key=Path("K"), known_hosts=Path("C:/hg/kh")) == [
        "ssh", "-i", "K", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=C:/hg/kh", "ameer@desktop-43dp1ti", n.notice_command(body)]
    assert n.MAX_B64 == 7000
    with pytest.raises(n.NoticeTooLong):
        n.notice_command({**body, "message": "y" * 6000})


def test_the_box_reply_is_read_tolerantly():
    from home_guard_project.fleet_contract.notices import notice_reply

    assert notice_reply(0, '{"result":"added","id":42}\r\n') == ("delivered", {"result": "added", "id": 42})
    assert notice_reply(0, 'noise\n{"result":"unchanged","id":42}\n')[0] == "delivered"
    assert notice_reply(1, '{"error":"bad base64"}') == ("rejected", {"error": "bad base64"})
    assert notice_reply(255, "")[0] == "retry"  # ssh could not connect
    assert notice_reply(None, "")[0] == "retry"  # timed out
    assert notice_reply(1, "Traceback ...\nModuleNotFoundError")[0] == "retry"  # the box has no notices command yet
    assert notice_reply(2, "usage: box [-h] ...")[0] == "retry"
    assert notice_reply(0, "")[0] == "retry" and notice_reply(0, '{"error":"x"}')[0] == "retry"


def test_host_keys_are_pinned_in_a_dedicated_file_never_the_founders_known_hosts():
    from pathlib import Path

    from home_guard_project.fleet_contract import notices as n

    body = {"schema_version": 1, "id": 1, "kind": "chat"}
    assert n.notice_known_hosts() == Path.home() / ".homeguard" / "box_known_hosts"
    argv = n.notice_ssh_argv("ameer", "box", body)
    options = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    assert options == ["BatchMode=yes", "ConnectTimeout=15", "StrictHostKeyChecking=accept-new",
                       "UserKnownHostsFile=" + (Path.home() / ".homeguard" / "box_known_hosts").as_posix()]
    assert not any("/.ssh/known_hosts" in a.replace("\\", "/") for a in argv)
    spaced = n.notice_ssh_argv("ameer", "box", body, known_hosts=Path("C:/Users/A B/.homeguard/box_known_hosts"))
    assert 'UserKnownHostsFile="C:/Users/A B/.homeguard/box_known_hosts"' in spaced


def test_a_changed_host_key_is_told_apart_from_an_unreachable_box():
    from home_guard_project.fleet_contract.notices import notice_host_key_changed

    changed = ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
               "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n"
               "Host key for desktop-43dp1ti has changed and you have requested strict checking.\n"
               "Host key verification failed.\n")
    assert notice_host_key_changed(255, changed)
    assert notice_host_key_changed(255, "Host key verification failed.\r\n")
    assert not notice_host_key_changed(255, "ssh: connect to host box port 22: Connection timed out")
    assert not notice_host_key_changed(None, "timed out") and not notice_host_key_changed(0, changed)
