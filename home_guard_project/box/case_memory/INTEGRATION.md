# Case memory: how to wire it in

The package is standalone and fully tested (`tests/box/test_case_memory_*.py`). Nothing in the guard loop or
the assistant calls it yet. This page lists the calls each side makes. Design: plan page section 6 and
`knowledge_base_home_gaurd/reports/זיכרון מקרים למתחקר.md`.

## What memory never does (enforced in code, tested)

- It never touches `[call_owner]` or an `escalation` label. Memory isn't even consulted for them.
- The veto means memory isn't consulted: an S/E category, a risk flag (`touching_handle`, `tool_in_hand`,
  `item_carried_away`, `face_covered`, `flashlight`, `crouching`, `running`, `weapon_visible`),
  `serious_behaviour`, or an incident on 2+ cameras at night, while asleep or while away.
- It never softens a `suspicious` label. A matching case only adds a context line.
- It lowers at most one step (alert → quiet → digest), and only for a case the owner confirmed. A case starts
  in shadow, where it still alerts.
- Only owner actions strengthen a case. Automatic matches are logged and only update `last_seen_at`.
- It never deletes history: delete, "moved away" and merge set `invalid_at` plus a reason.
- If it fails inside (exception, judge timeout, invalid judge answer, no key), the result is `("alert", None)`,
  which is today's behaviour.

## 1. Guard loop (`inference.py`, owner: the inference session)

**Startup (`run()`), once:**

```python
from .case_memory import configure, make_default
configure(make_default())   # <live_dir>/.registry/cases.jsonl, <live_dir>/.alert_embeddings.json, gpt-6-luna judge
```

Without `OPENAI_API_KEY` there's no embedder and no judge. High-band matches still work, and the middle band
alerts.

**In `_worker`, after `final_label` and `vlm_confirms`, before `dispatch_alert`:**

Skip this when the camera is muted, out of window or a false positive. Those paths return earlier and must stay
untouched.

```python
from .case_memory import CaseEvent, apply_case_memory
event = CaseEvent.build(
    event_id=job.stem, camera=camera_name, ts=alert_ts,
    observation=parsed,                       # the Eye's answer, as is
    tracker=job.tracker_facts or None,        # tracker.py: time_in_view_s, path, entry_edge, exit_edge
    situation=situation or None,              # situation.py: phase, house_state
    label=label, cameras_in_incident=incident_cameras or 1,
    eye_model=backend.model_name, prompt_version=PROMPT_VERSION)
level, note = apply_case_memory(event, {"final_label": label, "alert_command": cmd,
                                        "serious_behaviour": decision["serious_behaviour"]})
```

| `level` | What the guard loop does |
|---|---|
| `alert` | Today's path, unchanged. If `note` is set (`shadow`, `similar`, `context`, `keep_alerting`), add `note.text(lang)` as a line in the graded text, and send `note.buttons` as inline buttons. |
| `quiet` | Dispatch with `silent=True`. Add `note.text(lang)` ("Normal (per your explanation from 5.10): ...") and the "Not them" button. |
| `digest` | Don't dispatch. Write the event to the daily digest with `note.digest(lang)` and the clip link. |

Always record the outcome. Set `job.alert["delivery_level"] = level` and
`job.alert["case_memory"] = note.record() if note else None`. Also set
`job.alert["case_signature"] = event.signature.to_dict()`, which the assistant needs for the interview and the
buttons.

Notes for the integrator:

- Facts (`final_label`) run first. Memory sees the label after facts and priors.
- Today a `normal` delivery is already silent (`is_silent`). So on the box, `quiet` changes the wording
  ("normal per your explanation") but not the sound. The real saving is `digest`. Decide with the owner whether
  `quiet` should mean something stronger. Everything is in `ladder.py`.
- Until `tracker.py` lands, events have no path. The high band requires a path (`high_requires=("path",)`), so
  every match goes to the judge. This is deliberate.
- Gates fail closed. If a case has a path or a time-in-view limit and the event has none (the tracker is down),
  it doesn't match. If a case has a category and the event has none (old Eye), it doesn't match.

## 2. Assistant (brain/ Telegram, owner: the assistant-v2 session)

Every write takes `by=<owner id>`. Call these only after the existing verified-owner check. Group members who
aren't the owner can't create, confirm or widen anything (this is the defence against memory poisoning).

| Trigger | Call |
|---|---|
| Owner marks an alert "normal" / "expected" | `interview, step = interviewer.start(alert_id, Signature.from_dict(alert["case_signature"]))`. Show `step.text(lang)`, with `step.options` as buttons (always including skip / don't know). Store `interview.to_dict()` per chat. |
| Button or typed text during an interview | `step = interview.answer(code, text)`. Typed or voice text goes in `text` with `code=""`. Show `Question` and `SummaryCard` steps. On an `Outcome`, call `keeper.save_interview(store, outcome, by, embed)`. |
| `save_interview` returns `merge_proposed` | Show `result.text_he` (or `text_en`) with [combine] [keep separate], then call `keeper.confirm_merge(store, result, by, combine)`. |
| Alert button `confirm` (shadow "yes") | `keeper.on_button(store, "confirm", case_id, by, event_id=alert_id, example=Example(alert_id, sig, embedding))`. |
| Alert button `not_them` | `keeper.on_button(store, "not_them", case_id, by, event_id=alert_id)`. This steps the case back one stage and records a negative. |
| Alert button `keep_alerting` | `keeper.on_button(store, "keep_alerting", case_id, by)`. The case is kept for context only. |
| Alert button `fine_too` (similar but different) | `proposal = keeper.widen_for_event(store, case_id, sig, by)`. Show `keeper.describe_diff(proposal["diff"])` with yes/no, then call `store.answer_widen(proposal["wid"], yes, by)`. |
| Owner correction in chat ("only on weekdays") | `keeper.correct(store, case_id, by, weekdays=[...])` returns `narrowed` (applied now) or `proposed` (needs a yes, as above). |
| "What do you remember about the gate?" | `keeper.remember_listing(store, camera)`. Show each item with its delete button, which goes to `keeper.on_button(store, "delete", ...)`. |
| Daily | For each item in `keeper.due_reviews(store)`: send it, then call `store.ask_review(case_id)`. The buttons `keep` / `forget` go to `on_button`. |
| Nightly (Historian) | `routines.propose(records, store)` takes event records from the last 28 days (decision record plus observation / tracker / situation, or a stored `signature`). For each proposal, send `text_he` with yes/no, then call `routines.answer(store, proposal, yes, by, embed)`. |
| Situation building | `store.expecting_for(camera, ts)` sets `taxonomy.Context(expecting=True)`. |

`store` here is `case_memory.current().store`, or `CaseStore.at(default_path())`. Button callbacks need to be
encoded compactly (for example `cm:<action>:<case_id>`, under Telegram's 64-byte limit). Look up the alert id
from the message map the assistant already keeps.

## 3. Moving into facts.py (after `assistant-v2` merges)

Implement `CaseBackend` (two methods) over the facts journal:

```python
class FactsCaseBackend:
    def read_events(self):  # facts.jsonl lines whose event starts with "case.", with the prefix stripped
    def append(self, event):  # FactStore._append("case." + event["event"], **rest) under facts._FILE_LOCK
```

Then call `CaseStore(FactsCaseBackend(...))`. Case ids (`C…`), expecting ids (`X…`) and widening ids (`W…`)
don't collide with facts (`F…`) or proposals (`P…`). The nightly routine proposer is meant to replace the
facts' 10-hour proposal band (`PROPOSAL_BAND_HOURS`) for anything that silences. Expecting notes should then
move to the facts' planned `expecting` store.

## 4. Knobs (`CaseMemoryConfig`, `TrustLadder`): starting guesses, to be tuned on labelled pairs

| Knob | Default |
|---|---|
| Score weights | path 0.35, description 0.25, hour 0.20, time in view 0.10, appearance 0.10 |
| Bands | high ≥ 0.85 (and needs a path), middle ≥ 0.60, otherwise low |
| Hour gate | case window ± 15 min, wrapping past midnight |
| Path gate | same entry and exit zone, normalised edit distance ≤ 0.34 |
| Time in view | ≤ 2 × the longest example (at least +10 s) |
| Judge | gpt-6-luna, 8 s timeout. Sees the top 3 cases, or all gated cases when the camera has ≤ 5 live cases. |
| Trust ladder | 3 confirmations → quiet. 10 with no "not them" → digest (only if the owner chose digest). |
| Interview default window | event ± 30 min, rounded outward to 10 min. "Other hours too" uses ± 90 min. |
| Forgetting | 30 days unseen → "still relevant?" (the case pauses until answered) |
| Routines | 28 days of history. Proposed when seen on 6 of the last 14 days within 30 min. |

## 5. Evaluation

```python
from home_guard_project.box.case_memory import evaluation
report = evaluation.evaluate(memory, evaluation.make_pairs(case))     # plus real pairs built as Pair(...)
```

The report contains:

- `false_silence`: the rate, with a one-sided exact 95% upper bound. 0 of 100 is still up to about 3%, and 299
  clean pairs are needed to show it's under 1%.
- `saved_repeats`, `per_level` and `failures`.
- `operating_point`: the score threshold and the share of repeats saved at ≤ 1% false silence.

`similar_clothing` pairs are reported separately as `allowed_matches`. Memory recognises situations, not
people, and that residual risk is declared, not hidden. The `eval_prompt.py --pairs` mode in the plan can build
real pairs (owner-confirmed repeats, UCF-Crime segments) and call `evaluate`.
