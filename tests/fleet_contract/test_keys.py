from home_guard_project.fleet_contract.keys import parse_key, stem_kind, normalize_rel
import pathlib

def test_production_meta():
    k = parse_key("production_test/meta/left_side_1/2026-10-03/left_side_1_1791019693_alert.meta.json")
    assert (k.root, k.site, k.area, k.camera, k.day, k.stem, k.kind) == (
        "production", "test", "meta", "left_side_1", "2026-10-03", "left_side_1_1791019693_alert", "alert")

def test_site_with_underscores_and_fp():
    k = parse_key("dataset_ameer_batch_3/meta/back_door/2026-10-03/back_door_1791013299_fp.meta.json")
    assert k.site == "ameer_batch_3" and k.kind == "false_positive" and k.camera == "back_door"

def test_status_and_feedback_general():
    assert parse_key("dataset_test/_status/heartbeat.json").area == "status"
    f = parse_key("production_test/feedback/_general/2026-10-03/general_1791013449968.feedback.json")
    assert f.area == "feedback" and f.camera == "_general"

def test_yolo_label_and_crop_frames():
    k = parse_key("dataset_ameer_house/yolo/labels/back_door/2026-10-02/back_door_1790944263_trigger_f0002.txt")
    assert k.area == "yolo_labels" and k.stem == "back_door_1790944263_trigger"
    c = parse_key("dataset_test/vlm_crops/front_side/2026-10-03/front_side_1791020177_alert_f1.jpg")
    assert c.area == "vlm_crops" and c.stem == "front_side_1791020177_alert"

def test_outside_prefix():
    assert parse_key("tagging/x.json") is None and parse_key("dataset_uca") is None

def test_stem_kind():
    assert stem_kind("test_ch6_1790979739_alert") == ("test_ch6", 1790979739, "alert")
    assert stem_kind("weird") == ("weird", None, None)

def test_normalize_rel():
    assert normalize_rel("clips\\a\\2026-10-03\\x.mp4") == "clips/a/2026-10-03/x.mp4"
    assert normalize_rel("..\\..\\dataset_other\\x.mp4") is None
    assert normalize_rel("C:\\x.mp4") is None and normalize_rel("/x.mp4") is None

def test_every_fixture_listing_key_parses():
    for name in ["production_test.txt", "production_bian.txt", "dataset_test.txt"]:
        for line in pathlib.Path("tests/fleet_contract/fixtures", name).read_text().splitlines():
            key = line.split()[-1]
            assert parse_key(key) is not None, key


def test_raw_response_and_yolo_camera():
    k = parse_key("dataset_site/responses/back_door/2026-10-03/back_door_1791013299_fp.model_raw.txt")
    assert (k.stem, k.kind, k.ext) == ("back_door_1791013299_fp", "false_positive", ".txt")
    k = parse_key("dataset_site/yolo/images/back_door/2026-10-03/back_door_1791013299_random_f0000.jpg")
    assert (k.area, k.camera, k.day, k.kind) == ("yolo_images", "back_door", "2026-10-03", "random")


def test_unknown_area_preserves_original_key():
    key = "dataset_site/unrecognised/a.tmp"
    k = parse_key(key)
    assert k.key == key and k.area == "other" and k.camera is None
    assert k.stem == "a" and k.ext == ".tmp"


def test_normalize_dots_and_windows_roots():
    assert normalize_rel("././clips/a/./x.mp4") == "clips/a/x.mp4"
    for path in ["clips/../x.mp4", "C:x.mp4", "./C:/x.mp4", "\\\\server\\share\\x", "./../x"]:
        assert normalize_rel(path) is None, path


def test_stem_kind_with_underscores_and_unknown_kind():
    assert stem_kind("back_door_1791013299_fp") == ("back_door", 1791013299, "false_positive")
    assert stem_kind("back_door_1791013299_custom") == ("back_door", 1791013299, None)
    assert stem_kind("back_door_bad_alert") == ("back_door_bad_alert", None, "alert")



def test_parse_key_rejects_traversal_and_bad_layouts():
    good = "production_test/meta/front_side/2026-10-03/front_side_1791020177_alert.meta.json"
    assert parse_key(good) is not None
    for key in [
        "production_test/meta/front_side/2026-10-03/../front_side_1791020177_alert.meta.json",
        "production_test/meta/../../dataset_other/meta/a/2026-10-03/a_1_alert.meta.json",
        "production_test/meta/front_side\\2026-10-03/front_side_1791020177_alert.meta.json",
        "production_test/meta/front_side//front_side_1791020177_alert.meta.json",
        "production_test/meta/front_side/2026-10-03/",
        "production_test/meta/front_side/2026-10-03/extra/front_side_1791020177_alert.meta.json",
        "production_test/meta/front_side/front_side_1791020177_alert.meta.json",
        "production_test/clips/front_side/2026-10-03" + "9" * 300 + "/front_side_1791020177_alert.mp4",
        "production_test/clips/front_side/26-10-03/front_side_1791020177_alert.mp4",
        "production_test/clips/front side!/2026-10-03/x_1_alert.mp4",
        "production_test/clips/" + "c" * 81 + "/2026-10-03/x_1_alert.mp4",
        "production_test/clips/./2026-10-03/x_1_alert.mp4",
        "production_test/clips/../2026-10-03/x_1_alert.mp4",
        "dataset_test/yolo/labels/back_door/2026-10-02/extra/x_f0000.txt",
        "dataset_test/_status/sub/heartbeat.json",
        "dataset_test/_status/",
        "dataset_test/unrecognised/../a.tmp",
        "dataset_test//a.tmp",
    ]:
        assert parse_key(key) is None, key


def test_parse_key_accepts_the_general_feedback_camera_and_dotted_names():
    assert parse_key("production_test/feedback/_general/2026-10-03/general_1.feedback.json").camera == "_general"
    assert parse_key("production_test/clips/cam.v2-a/2026-10-03/cam.v2-a_1_alert.mp4").camera == "cam.v2-a"
