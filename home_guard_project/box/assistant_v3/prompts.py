# -*- coding: utf-8 -*-
"""The three prompts of v3, each small and with one job (report §8.2): UNDERSTAND (acts JSON), the WRITER persona
(the reply, with or without tools), and the CRITIC (the pre-send check). Behaviour is taught by EXAMPLES, not by a
growing list of rules: a new failure becomes a golden case and, when the model misread or mis-phrased, one more
example here - never a new paragraph. The words live in home_guard_project/prompts/assistant_v3_*.

The examples are written for this file. They are modelled on the kinds of turns in the owner's real chats but use
other people, places and words (gardeners, the neighbours' yard, the storeroom), so they teach the pattern and not
the golden suite's answers.
"""

from ...prompts import load

UNDERSTAND = load("assistant_v3_understand.system_prompt")
WRITER = load("assistant_v3_writer.system_prompt")
TOOLS_NOTE = load("assistant_v3_tools_note.prompt")      # {{budget}}: fill(TOOLS_NOTE, budget=...)
CRITIC = load("assistant_v3_critic.system_prompt")
SUMMARY = load("assistant_v3_summary.system_prompt")    # the older-turns summary (agent._summarize)
