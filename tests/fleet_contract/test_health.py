from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from home_guard_project.fleet_contract.health import camera_stale, verdict
from home_guard_project.fleet_contract.legacy import parse_heartbeat


def load(name):
    return json.loads((Path("tests/fleet_contract/fixtures") / name).read_text(encoding="utf-8"))


NOW = datetime(2026, 10, 3, 11, 15, 14, tzinfo=timezone.utc)


@pytest.fixture
def hb():
    return parse_heartbeat(load("heartbeat.json"))


def test_no_heartbeat():
    status, reasons = verdict(None, NOW)
    assert status == "unknown" and reasons[0]["code"] == "no_heartbeat"


def test_old_heartbeat(hb):
    status, reasons = verdict(replace(hb, time_utc=NOW - timedelta(hours=3)), NOW)
    assert status == "offline"
    reason = next(r for r in reasons if r["code"] == "heartbeat_old")
    assert reason["message"] == "Last heard 3 h ago" and reason["severity"] == "offline"


def test_heartbeat_without_valid_time_is_offline(hb):
    assert verdict(replace(hb, time_utc=None), NOW) == ("offline", [{
        "code": "heartbeat_old", "message": "Heartbeat has no valid time", "severity": "offline",
    }])


def test_stopped_by_owner(hb):
    status, reasons = verdict(replace(hb, stopped=True, collector_running=False), NOW)
    assert status == "warning" and [r["code"] for r in reasons] == ["stopped_by_owner"]


def test_engine_down(hb):
    status, reasons = verdict(replace(hb, collector_running=False), NOW)
    assert status == "critical" and reasons[0]["code"] == "engine_down"


def test_disk_critical(hb):
    status, reasons = verdict(replace(hb, disk_free_gb=10), NOW)
    assert status == "critical" and reasons[0]["severity"] == "critical"
    assert "10" in reasons[0]["message"]


def test_disk_warning(hb):
    status, reasons = verdict(replace(hb, disk_free_gb=30), NOW)
    assert status == "warning" and reasons[0]["severity"] == "warning"


def test_camera_quiet(hb):
    status, reasons = verdict(replace(hb, cameras={
        "front_side": NOW - timedelta(hours=26, minutes=59),
        "back_door": NOW - timedelta(hours=25),
        "recent": NOW - timedelta(hours=23),
    }), NOW)
    assert (status, reasons) == ("warning", [{
        "code": "camera_quiet", "message": "No clip for 25-26 h from Front side, Back door — check they have power and network",
        "severity": "warning",
    }])


def test_camera_never(hb):
    assert verdict(replace(hb, cameras={"left_side_2": None}), NOW) == ("warning", [{
        "code": "camera_never", "message": "No clip recorded yet from Left side 2 — check its login and stream on the box",
        "severity": "warning",
    }])


def test_upload_backlog(hb):
    status, reasons = verdict(replace(hb, clips_outbox=501), NOW)
    assert status == "warning" and reasons[0]["code"] == "upload_backlog"


def test_healthy(hb):
    assert verdict(hb, NOW) == ("healthy", [])


def test_worst_wins(hb):
    status, reasons = verdict(replace(hb, stopped=True, collector_running=False, disk_free_gb=10), NOW)
    assert status == "critical"
    assert {r["severity"] for r in reasons} == {"warning", "critical"}
    assert "engine_down" not in {r["code"] for r in reasons}


def test_reasons_are_human(hb):
    status, reasons = verdict(replace(hb, cameras={"front_side": None, "back_door": NOW - timedelta(days=2)}), NOW)
    assert status == "warning"
    assert reasons == [
        {"code": "camera_quiet", "message": "No clip for 48 h from Back door — check it has power and network", "severity": "warning"},
        {"code": "camera_never", "message": "No clip recorded yet from Front side — check its login and stream on the box",
         "severity": "warning"},
    ]


@pytest.mark.parametrize("age,expected", [
    (None, True), (timedelta(hours=24), True),
    (timedelta(hours=24) - timedelta(microseconds=1), False),
    (timedelta(hours=25), True), (timedelta(hours=-1), False),
])
def test_camera_stale_boundaries(age, expected):
    newest = NOW - age if age is not None else None
    assert camera_stale(newest, NOW) is expected


@pytest.mark.parametrize("minutes,expected", [(90, "healthy"), (90.01, "offline"), (-1, "healthy")])
def test_heartbeat_age_boundaries(hb, minutes, expected):
    assert verdict(replace(hb, time_utc=NOW - timedelta(minutes=minutes)), NOW)[0] == expected


@pytest.mark.parametrize("gb,expected", [(19.99, "critical"), (20, "warning"), (49.99, "warning"), (50, "healthy"), (None, "healthy")])
def test_disk_boundaries(hb, gb, expected):
    assert verdict(replace(hb, disk_free_gb=gb), NOW)[0] == expected


def test_backlog_boundary(hb):
    assert verdict(replace(hb, clips_outbox=500), NOW) == ("healthy", [])


@pytest.mark.parametrize("time_utc", [None, NOW - timedelta(hours=3)])
def test_offline_wins_and_retains_other_reasons(hb, time_utc):
    status, reasons = verdict(replace(hb, time_utc=time_utc,
        collector_running=False, disk_free_gb=5, clips_outbox=900, cameras={"front_side": None}), NOW)
    assert status == "offline" and len(reasons) == 5
    assert {r["code"] for r in reasons} == {"heartbeat_old", "engine_down", "disk_low", "camera_never", "upload_backlog"}


def test_unknown_engine_is_not_false(hb):
    assert verdict(replace(hb, collector_running=None, stopped=None), NOW) == ("healthy", [])


def test_naive_and_offset_datetimes(hb):
    assert verdict(hb, NOW.replace(tzinfo=None)) == ("healthy", [])
    assert not camera_stale(NOW.replace(tzinfo=None), NOW)
    assert camera_stale((NOW - timedelta(hours=24)).astimezone(timezone(timedelta(hours=3))), NOW)
    for time_utc in [NOW.replace(tzinfo=None), NOW.astimezone(timezone(timedelta(hours=3)))]:
        assert verdict(replace(hb, time_utc=time_utc), NOW) == ("healthy", [])
    status, reasons = verdict(replace(hb, cameras={
        "front_side": (NOW - timedelta(hours=26, minutes=59)).replace(tzinfo=None),
        "back_door": (NOW - timedelta(hours=25)).astimezone(timezone(timedelta(hours=3))),
    }), NOW.replace(tzinfo=None))
    assert status == "warning" and reasons[0]["message"].startswith("No clip for 25-26 h from Front side, Back door")


def test_alert_hours_do_not_change_the_prescribed_rules(hb):
    assert verdict(replace(hb, collector_running=False), NOW, alert_hours=(20, 6))[0] == "critical"


def test_camera_names_are_never_raw_ids(hb):
    status, reasons = verdict(replace(hb, cameras={"ameer_tes2_ch6": NOW - timedelta(hours=30)}), NOW)
    assert reasons[0]["message"] == "No clip for 30 h from Camera 6 — check it has power and network"


def test_renamed_site_ids_are_retired_and_never_warn(hb):
    from home_guard_project.fleet_contract.health import split_cameras
    cameras = {"ameer_tes2_ch6": NOW - timedelta(days=5), "ameer_tes2_ch3": NOW - timedelta(days=6),
               "ameer_week_0_1_ch6": NOW - timedelta(minutes=5), "ameer_week_0_1_ch2": NOW - timedelta(hours=30),
               "ameer_week_0_1_ch4": None}
    current, retired = split_cameras(cameras, NOW, site="ameer_week_0_1")
    # ch6 moved to the new id; ch3 sits under the old site's name; the quiet ch2 and the silent ch4 of this site stay
    assert current == ["ameer_week_0_1_ch2", "ameer_week_0_1_ch4", "ameer_week_0_1_ch6"]
    assert retired == ["ameer_tes2_ch3", "ameer_tes2_ch6"]
    status, reasons = verdict(replace(hb, cameras=cameras), NOW, cameras=current,
                              names={"ameer_week_0_1_ch2": "כניסה ראשית"})
    assert [r["message"] for r in reasons] == [
        "No clip for 30 h from כניסה ראשית — check it has power and network",
        "No clip recorded yet from Camera 4 — check its login and stream on the box"]
    assert "tes2" not in repr(reasons) and "Ameer" not in repr(reasons)


def test_a_recent_event_keeps_a_camera_current_and_the_box_list_wins():
    from home_guard_project.fleet_contract.health import split_cameras, listed_cameras, camera_list
    cameras = {"old_site_ch1": NOW - timedelta(days=3), "front_door": NOW - timedelta(days=9)}
    assert split_cameras(cameras, NOW, site="new_site", recent=["old_site_ch1"]) == (["front_door", "old_site_ch1"], [])
    assert split_cameras(cameras, NOW, site="new_site") == (["front_door"], ["old_site_ch1"])
    # the box's heartbeat camera_list (beelink-collector-box 79e9785); `cameras` stays the clip times
    body = {"cameras": {"x": {}}, "camera_list": [
        {"id": "new_site_ch1", "name": " Gate ", "channel": "1", "enabled": True},
        {"id": "new_site_ch2", "name": "", "channel": "2", "enabled": False}, {"name": "no id"}, "junk"]}
    assert camera_list(body) == [{"id": "new_site_ch1", "name": "Gate", "enabled": True},
                                 {"id": "new_site_ch2", "name": "", "enabled": False}]
    assert listed_cameras(body) == {"new_site_ch1": "Gate"}
    assert listed_cameras({"cameras": {"x": {}}}) is None and listed_cameras(None) is None
    assert split_cameras(cameras, NOW, listed=["new_site_ch1"]) == (["new_site_ch1"], ["front_door", "old_site_ch1"])


def test_an_old_site_id_takes_the_owner_name_by_its_channel():
    from home_guard_project.fleet_contract.health import camera_label, owner_name
    names = {"ameer_week_0_1_ch6": "כניסה ראשית", "ameer_week_0_1_ch3": "פרגולה"}
    assert camera_label("ameer_tes2_ch6", names) == "כניסה ראשית" and owner_name("ameer_tes2_ch3", names) == "פרגולה"
    assert camera_label("ameer_tes2_ch2", names) == "Camera 2" and owner_name("ameer_tes2_ch2", names) == ""
    # two current cameras on one channel: ambiguous, the box's rule shows none rather than guess
    assert camera_label("old_ch6", {**names, "other_ch6": "Gate"}) == "Camera 6"
