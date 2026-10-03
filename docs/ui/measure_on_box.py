"""Read-only installer probe. Run locally on the box; never opens a connection."""
import argparse
import json
from pathlib import Path
import time


def read_metrics(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not 0 <= time.time() - data["updated"] <= 3:
        raise RuntimeError("Publisher telemetry is stale; run the updated inference_preview engine.")
    if not data.get("detector_instrumented"):
        raise RuntimeError("Detector-loop counter is unavailable; use inference_preview mode.")
    return data


def summarize(first, last):
    if (first["pid"], first["started"]) != (last["pid"], last["started"]):
        raise RuntimeError("Engine restarted during the sample; repeat the measurement.")
    elapsed = last["monotonic"] - first["monotonic"]
    if elapsed <= 0:
        raise RuntimeError("Telemetry clock did not advance.")
    def rate(key): return (last[key] - first[key]) / elapsed
    return {
        "seconds": round(elapsed, 2),
        "publisher_one_core_percent": round(100 * rate("publisher_cpu_seconds"), 2),
        "detector_loop_hz": round(rate("detector_loops"), 3),
        "hero_published_fps": round((last["published_roles"]["hero"]-first["published_roles"]["hero"])/elapsed, 3),
        "thumbnail_published_fps_total": round((last["published_roles"]["thumbnail"]-first["published_roles"]["thumbnail"])/elapsed, 3),
        "per_camera_published_fps": {name: round((count-first["published"].get(name, 0))/elapsed, 3)
                                     for name, count in last["published"].items()},
        "selected_hero_thumbnail_fps": last["rates"],
        "cpu_cap_percent": last["cap_percent"],
    }


def sample(path, visible):
    first = read_metrics(path)
    if first["visible"] != visible:
        raise RuntimeError("Window visibility does not match this sample; wait for the lease to update.")
    last = first
    floor_over_budget = False
    while last["monotonic"] - first["monotonic"] < 60:
        time.sleep(1)
        last = read_metrics(path)
        if last["visible"] != visible:
            raise RuntimeError("Window visibility changed during the sample; repeat it.")
        if (last["pid"], last["started"]) != (first["pid"], first["started"]):
            raise RuntimeError("Engine restarted during the sample; repeat it.")
        floor_over_budget |= last.get("floor_over_budget", False)
    result = summarize(first, last)
    result["floor_over_budget_observed"] = floor_over_budget
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", required=True, type=Path, help="Local LOG_DIR/preview directory")
    args = parser.parse_args()
    path = args.preview_dir / "publisher_metrics.json"
    results = {}
    for label, visible in (("closed", False), ("open", True)):
        action = "Close the app (leave detection running)" if not visible else "Open the Live dashboard and keep the same hero and cameras visible"
        input(f"{action}. Wait 15 seconds for rates/lease to settle, then press Enter for a 60 s sample: ")
        print(f"Measuring app {label} for 60 s...", flush=True)
        results[label] = sample(path, visible)
        print(json.dumps({label: results[label]}, indent=2), flush=True)
    closed = results["closed"]["detector_loop_hz"]
    results["detector_open_vs_closed_percent"] = round((results["open"]["detector_loop_hz"]/closed-1)*100, 2) if closed else None
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"Measurement unavailable: {exc}")
