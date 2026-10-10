"""Shared test setup: the AI usage ledger writes nothing unless a test points it at a temporary folder, and an
inference run() in a test starts no system notices (no alive file on the developer's disk, no Telegram)."""
import os

os.environ["HOMEGUARD_USAGE_LEDGER"] = "off"
os.environ["HOMEGUARD_SYSTEM_NOTICES"] = "off"
