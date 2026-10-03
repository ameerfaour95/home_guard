"""Setup contract: replace only Backend to integrate the real installer later."""

from dataclasses import dataclass, field
from typing import Protocol

STEPS = ("update", "name_step", "network_step", "camera_step", "readiness")


@dataclass
class Answers:
    address: str = ""
    network: str = "ethernet"
    ssid: str = ""
    wifi_password: str = field(default="", repr=False)
    house: str = ""
    owner_name: str = field(default="", repr=False)
    owner_phone: str = field(default="", repr=False)
    installer: str = ""
    consent_live: bool = False
    consent_recordings: bool = False
    consent_training: bool = False
    show_cameras: bool = False
    alerts: bool = False
    start_hour: int = 0
    end_hour: int = 0
    cooldown_sec: int = 120
    find_cameras: bool = True
    camera_user: str = ""
    camera_password: str = field(default="", repr=False)


@dataclass(frozen=True)
class Check:
    status: str
    message_key: str


@dataclass(frozen=True)
class Result:
    step: str
    status: str
    message_key: str
    checks: tuple[Check, ...] = ()
    rescue_name: str = ""
    rescue_password: str = field(default="", repr=False)


class Backend(Protocol):
    def execute(self, step: str, answers: Answers) -> Result: ...


class SimulatedBackend:
    delay_ms = 1600
    delays = dict(zip(STEPS, (2400, 900, 2200, 4200, 1400)))

    def delay_for(self, step):
        return self.delays[step]

    def __init__(self, failure=False):
        self.failure = failure

    def execute(self, step, answers):
        if step == "network_step" and self.failure:
            return Result(
                step,
                "FAIL",
                "network_fail" if answers.network == "wifi" else "network_cable_fail",
            )
        if step == "camera_step" and not answers.find_cameras:
            return Result(step, "WARN", "skipped")
        if step == "readiness":
            checks = (
                Check("PASS", "check_collector"),
                Check("PASS", "check_network"),
                Check("PASS", "check_folder"),
                Check("WARN", "check_power"),
            )
            return Result(step, "WARN", "ready_warn", checks)
        return Result(
            step,
            "PASS",
            dict(zip(STEPS, ("updated", "named", "network_ok", "found", "ready")))[
                step
            ],
        )


class Sequence:
    def __init__(self, backend, answers):
        self.backend, self.answers = backend, answers
        self.results = []
        self.index = 0
        self.failed = False

    @property
    def done(self):
        return self.failed or self.index == len(STEPS)

    def advance(self):
        if self.done:
            return None
        step = STEPS[self.index]
        try:
            result = self.backend.execute(step, self.answers)
            if result.step != step or result.status not in ("PASS", "WARN", "FAIL"):
                raise ValueError("Invalid setup result")
        except Exception:
            result = Result(step, "FAIL", "step_error")
        self.results.append(result)
        self.index += 1
        self.failed = result.status == "FAIL"
        return result
