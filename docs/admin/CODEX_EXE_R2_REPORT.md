# HomeGuardAdmin.exe — Round 2

Implemented on `admin-console-ui` in `C:\Users\ameer\Ameer\home_guard_admin_ui`. No push, SSH, AWS access, or changes to another worktree. The two pre-existing run logs remain untracked.

## Delivered

- Fleet uses the single documented `SEVERITY` constant: offline, critical, warning, unknown, healthy. Equal-verdict rows retain the server's order. The reason column expands first, uses an inline verdict/reason layout at wide sizes, and numeric columns align right. The 24-hour strip consumes every events cursor, highlights alerts, marks owner false alarms, and exposes counts on hover.
- Rail and customer tabs have native line icons. Review has a badge calculated from unreviewed events in the last 24 hours; review mutations update it.
- Customer header shows identity, site, worst/selected box verdict and reason, consent flags, and last-heard age. Timeline is the default. Conversation, Config, and Access have designed coming-soon states.
- Timeline has UTC-hour buckets displayed in customer time, 6 h / 24 h / 7 d / custom ranges, density-cell filtering, every specified API filter, debounced search, virtualised rows, independent thumbnail loading, explicit and automatic cursor paging, and review/flag keyboard actions. Failed/missing thumbnails retain a camera tile. Range and filter changes discard late responses from previous requests.
- Event playback uses QMediaPlayer and QVideoSink, with a separate transparent widget for boxes. It draws the nearest saved frame, supports normalized coordinates and letterboxing, shows provenance, colors person/vehicle/animal by theme, and offers D, Space, frame steps, speed, scrubber, and ±500 ms offset. The overlay uses decoded-frame timestamps; positive offset selects later detections. Saved detections are sorted once; playback uses binary search.
- A filmstrip hover preview reads `fps`, `tile_w`, `tile_h`, and `count` from optional artifact `detail`; both horizontal and tiled sprites work. The demo omits a filmstrip artifact, as permitted; a Qt test exercises the preview with a generated sprite.
- AI record shows the decision and reason, summary, honest AI state, model/prompt version, input frames fetched through artifact access, parsed values, collapsible raw answer/full prompt with copy, and owner feedback history. Dispatch is admin/support only. Raw metadata is admin only. Event navigation stays within the timeline's filtered list and fetches another page at its end.
- Labelers enter through Review, which reuses the timeline/event workspace without calling customer or Fleet routes. Names/cameras are rendered exactly as supplied by the backend. Conversation, Access, dispatch, owner raw text, and raw metadata are hidden. AI evidence remains available. The fuller Review queue and Studio workflows remain for Round 3.
- HTTP methods now cover detections, review PATCH, and artifact-access POST; all share the one-refresh-on-401 path. Authenticated thumbnail redirects do not forward Bearer credentials to the media host. API calls, fixture reads, and media-byte fetches run in workers; QMediaPlayer handles its own asynchronous stream.

## Demo assets

`python -m home_guard_project.admin.make_demo_media` regenerates 96 events, matching detail/detection responses, input JPEGs, raw answers, and three synthetic clips. Requires cv2 and ffmpeg; the desktop runtime needs neither the generator nor a system ffmpeg executable.

- Clips: person, car, dog; 640×360, 12 fps, 6 seconds; H.264 with faststart.
- MP4 total: **33,528 bytes**.
- Media directory, including input frames/raw answers: **130,552 bytes**.
- Entire demo fixture directory, including existing Fleet/customer data and all generated JSON: **806,348 bytes**, below 1 MB.
- All three MP4s and nine JPEGs are force-added despite ignore rules.
- Fixtures include real, failed, fallback and absent AI, all box provenance states, review flags, feedback, an expired recording, and a missing thumbnail. Review changes persist for the DemoBackend session.

## File map

| Area | Files under `home_guard_project/admin/` |
|---|---|
| Customer workspace | `customer.py`, `timeline.py`, `timeline_model.py` |
| Player and record | `event_view.py`, `player.py`, `ai_record.py` |
| Pure math and wording | `event_logic.py` |
| Shared visual components | `widgets/activity.py`, `widgets/icons.py` |
| API/data | `models.py`, `backend.py`, `http_backend.py`, `demo_backend.py` |
| Demo production | `make_demo_media.py`, `demo_data/` |
| Fleet/shell/style | `fleet.py`, `fleet_model.py`, `shell.py`, `theme.py` |
| Verification | `screenshots_r2.py`, `__main__.py` smoke path; `tests/admin/test_round2.py` |

Qt Multimedia is supplied by `pyside6-addons` in the admin dependency group. OpenCV is in that group to make the demo generator reproducible in the small admin environment. No dependency was added to the main group. The renderer follows Qt's documented [QVideoSink frame interface](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QVideoSink.html).

## Verification

Prepared with `uv --system-certs sync --only-group admin`; tests do not access the network.

```powershell
uv run --group admin --no-sync pytest tests/admin -q
```

**262 passed.** `--no-sync` uses the prepared admin environment rather than installing the unrelated full box/ML dependency set. The equivalent `uv run --only-group admin pytest tests/admin -q` also passed.

Coverage includes all fixture schemas, stable Fleet ties, hour/range boundary math, letterboxing in both directions, normalized/clamped boxes, nearest-frame ties and offsets, provenance wording, AI-status wording, contract filter names, cursor completeness, auto paging, keyboard mutations, stale query responses, pseudonym camera filters, role hiding, real H.264 decode/seek and D toggle, expired media, filmstrip hover, HTTP mutation refresh, and redirect credential separation. Existing Round 1 tests continue to pass.

```powershell
home_guard_project/admin/build_admin_exe.ps1 -SkipSync
dist/HomeGuardAdmin/HomeGuardAdmin.exe --demo --smoke-test
dist/HomeGuardAdmin/HomeGuardAdmin.exe --smoke-test
```

**Build succeeded. Both packaged smoke tests exited 0.** The source demo smoke also exited 0. The demo smoke opens a customer and event, decodes and seeks the bundled clip to its middle, checks saved detections, and exits. The sign-in smoke checks startup and bundled icon loading. Both run offscreen.

Output: `dist/HomeGuardAdmin/HomeGuardAdmin.exe` — **5,446,450 bytes (5.19 MiB)**. Complete one-directory bundle: **144,753,969 bytes (138.05 MiB)**. Distribute the full directory. Build log: `build/admin-r2-build.log` (local, not committed).

## Screenshots

Generated with `python -m home_guard_project.admin.screenshots_r2`. All 12 screenshots were opened and visually inspected. The harness asserts the compact shell is actually 1366×768 and waits for a real decoded mid-clip frame. Inspection led to fixes for hidden-page minimum heights, text glyphs, native tab borders, overly large density rows, the Fleet gap, and overlay/transport alignment.

| Screenshot | Size / purpose |
|---|---|
| [r2-customer-timeline-1920.png](screenshots/r2-customer-timeline-1920.png) | 1920×1080, customer timeline |
| [r2-customer-timeline-1366.png](screenshots/r2-customer-timeline-1366.png) | 1366×768, compact timeline |
| [r2-timeline-filtered.png](screenshots/r2-timeline-filtered.png) | Hour/camera cell filter |
| [r2-event-boxes-on.png](screenshots/r2-event-boxes-on.png) | 1920×1080, decoded person frame and sampled boxes |
| [r2-event-1366.png](screenshots/r2-event-1366.png) | Compact player with pillarboxing |
| [r2-event-boxes-off.png](screenshots/r2-event-boxes-off.png) | Same event, overlay hidden |
| [r2-event-ai-failed.png](screenshots/r2-event-ai-failed.png) | Honest failure state and vehicle boxes |
| [r2-event-ai-fallback.png](screenshots/r2-event-ai-fallback.png) | Honest fallback state and recomputed animal boxes |
| [r2-labeler-event.png](screenshots/r2-labeler-event.png) | Pseudonyms and role hiding |
| [r2-fleet-activity.png](screenshots/r2-fleet-activity.png) | Wide Fleet, activity strip, reason allocation |
| [r2-nav-icons.png](screenshots/r2-nav-icons.png) | Rail icons and live Review count |
| [r2-timeline-light.png](screenshots/r2-timeline-light.png) | Light-theme check |

## Integration gaps and questions

1. **Artifact metadata:** Task 2's `ArtifactOut` omits `detail`, while Task 12 and this brief require it for filmstrips. The client accepts an optional `detail={}` extension and remains compatible when absent. Please include it in the server schema/OpenAPI before filmstrip integration.
2. **Labeler timezone:** EventSummary/Detail do not supply the customer's timezone, and the customer route is forbidden to labelers. Their review workspace explicitly shows UTC. Customer pages use `CustomerOut.timezone`. Can the event contract supply a timezone without exposing identity?
3. **Silent cameras:** The current contract has no camera catalog in CustomerOut. Density rows come from cameras represented by events in the selected range. To show a camera with no events at all, the server needs to supply its name/pseudonym separately.
4. **Large histories:** Density and the badge consume all relevant event cursors in workers because no aggregate route exists. This gives complete counts with the current contract; a server aggregation route would avoid fetching whole summaries for busy 7-day/custom ranges.
5. No live Cloud server was exercised. HTTP wire behavior is covered by MockTransport; real-service integration remains Round 4. Conversation/Config/Access are intentionally coming-soon states. Custom ranges are entered in explicitly labelled UTC and displayed in customer time, with a 31-day maximum.

## Commits

- `ce037ed` — Fleet order/activity, evidence backend, contract models, synthetic fixture generator and committed assets.
- `f1941dd` — Customer timeline, player, AI/feedback record, role-aware workspace, and offscreen behavioral tests.
- Final report/screenshots commit includes the last Fleet dot alignment correction and screenshot harness; see branch HEAD.

Every commit ends with `Co-Authored-By: Codex <noreply@openai.com>`.
