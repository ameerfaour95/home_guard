# Staged clips to film at home (eval "misses" set, part 2)

The frozen eval set (`home_guard_eval/eval_set_v2`) has only 4 alerts filmed by our own cameras. Everything else comes
from UCF-Crime, SmartHome-Bench and web videos, whose cameras, angles and houses look nothing like ours. This list is
the clips to stage at home so the eval can say how the box does **on our cameras**, on the hard cases: night, partial
occlusion, subtle attempts and loitering.

Film each scenario on the camera named, with the collector running as usual, so the clips arrive the normal way
(`dataset_multi/clips/<camera>/...`). Then tag them in Label Studio like any other clip (`[alert]` plus a plain
description). They go into a **new** eval folder (`eval_set_v3`); `eval_set_v2` is frozen and stays as it is.
**Keep every staged clip out of training** (they are test data): tag them in a batch of their own, e.g.
`ameer_house_staged_1`, so the dataset builder can hold the batch out.

## Before you start

- Tell the family and the neighbours. Do not stage anything that a passer-by or a monitoring company could report.
- No real damage and no real weapons. "Prying" is a wooden stick or the flat of a screwdriver against a folded cloth;
  nothing touches a lock or a frame hard enough to mark it.
- Two people: one acts, one keeps the log and watches the live view so every take is really in frame.
- Wear different clothes across takes (light top, dark hoodie, cap, a bag on the back). One actor in every take is
  fine; a second person in a few takes is better (the eval has many lone-intruder clips already).

## How to shoot one take

- 15 to 25 seconds per take. Walk in from outside the frame, so the trigger fires on arrival.
- Hold the act for at least 3 seconds. The eval takes its 5 frames across the act, using the start and end you log.
- The box itself asks the AI once, when it first sees a person (the 5 frames, one per second, up to that moment), and
  then not again on that camera for 2 minutes. So in about half the takes do the act **right as you come into view**:
  that is what the box really gets to see. In the other half, walk in, pause, then act.
- Leave at least 2 minutes between takes on the same camera (the box's per-camera wait), so each take is its own
  clip and its own AI call.
- Log every take on one line: `take id, camera, scenario, day/night, start time (hh:mm:ss), end of the act, notes`.
  The clip names carry the time, so the log is how each clip gets matched to its scenario and its act's start and end.

Night means after dark with the house lights as they normally are. Where the table says "IR", do the night takes with
the outside lights off, so the camera is in black-and-white infrared (the hardest case, and the one that matters most).

## Alerts to stage

| # | Scenario | Category | Camera | Day takes | Night takes | Variant to include |
|---|---|---|---|---|---|---|
| A1 | Walk to the front door, try the handle 2-3 times, look around, leave | S1 testing access | main_door | 3 | 3 (2 of them IR) | one take with the hood up |
| A2 | Kneel at the front door and "pry" at the lock or the frame for 5 s | E1 forced entry | main_door | 2 | 3 (IR) | one take half hidden behind the pillar or the plant |
| A3 | Side door: try the handle, then push on the ground-floor window and shutter | S1 testing access | left_side_1 | 3 | 3 (2 IR) | one take with a second person keeping watch |
| A4 | Look into a ground-floor window, hands cupped against the glass; at night with a phone torch | S2 looking in | left_side_1 | 2 | 3 (torch) | one take where only head and shoulders show above the wall line |
| A5 | At the gate: try the gate handle, push the gate, reach over or through it toward the latch | S1 testing access | front_side | 3 | 3 (2 IR) | one take standing so the gate bars cover half the body |
| A6 | Climb over the front wall or the gate into the driveway | E2 climbing in | front_side | 2 | 3 (IR) | one take at the very edge of the frame |
| A7 | Climb the side wall into the side yard | E2 climbing in | right_side, then left_side_2 | 2 each | 2 each (IR) | one take that drops behind the wall out of view |
| A8 | Loiter outside the gate for 60-90 s: stand, phone out, look at the house, walk off, come back | S4 lingering | front_side (main_door sees it too) | 3 | 3 | one take half behind the gate pillar |
| A9 | Walk slowly along the fence and stop to look over at each window | S3 surveying | right_side, then left_side_1 | 2 | 2 (IR) | |
| A10 | At the car in the driveway: try each car door, then look in with a torch | S1 / S2 | main_door, front_side | 3 | 3 (torch) | one take crouched at the far side of the car (car hides the legs and body) |
| A11 | Crouch and hide behind the car or the wall for 10 s, peek out, leave | S8 hiding | front_side, main_door | 2 | 3 (IR) | occluded by design |
| A12 | Walk up to the door with the face covered (hood plus scarf), stop, look around | S5 hiding the face | main_door | 2 | 2 | |
| A13 | Take a parcel from the doorstep and walk off with it | E3 theft | main_door | 2 | 1 | |
| A14 | Partial-occlusion repeats of A1, A5 and A10: only half the body visible (pillar, gate bars, car, plant) | S1 | main_door, front_side | 3 | 3 (IR) | the point of this row |

About 80 takes. Two evenings plus one morning is enough.

## Look-alike normals (film these too)

Without them the eval cannot tell a model that catches intruders from one that flags everyone at night.

| # | Scenario | Camera | Day takes | Night takes |
|---|---|---|---|---|
| N-A | Owner opens the gate with the key or remote and walks in | front_side | 2 | 2 (IR) |
| N-B | Owner unlocks the car with the key fob and gets something out of it | main_door, front_side | 2 | 2 (torch, like A10) |
| N-C | Courier walks to the door, leaves a parcel, rings, waits, leaves | main_door | 2 | 2 |
| N-D | Family member stands at the gate on the phone for 60 s (looks like A8) | front_side | 2 | 2 |
| N-E | Someone waters the plants along the side wall | right_side, left_side_1 | 2 | 1 |

About 20 takes.

## After filming

1. Check every take is in `dataset_multi/clips/<camera>/<date>/` and that the log matches the clip times.
2. Tag in Label Studio in a batch of its own (`ameer_house_staged_1`): `[alert]` and what happens, in plain words, for
   the A rows; a plain description for the N rows.
3. Once the batch is in `home_guard_dataset`, build `eval_set_v3`: copy `eval_set_v2/frames`, run `prepare`, set
   `category`, `subset: misses` and `hard` on the staged alerts (from the log), and `freeze`.
