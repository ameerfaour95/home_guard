"""Shared test setup: the AI usage ledger writes nothing unless a test points it at a temporary folder."""
import os

os.environ["HOMEGUARD_USAGE_LEDGER"] = "off"
