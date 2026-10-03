"""The owner's assistant, version 2 ("the brain").

Small units, one job each: ``i18n`` (every sentence the box writes, in three
languages), ``mode`` (Guard or Assistant), ``aliases`` and ``registry`` (what the
house looks like right now), ``receipts`` (proof of every action), ``memory``
(the conversation with its event handles), ``events`` (saved events and their
descriptions), ``vision`` and ``media`` (pictures and video), ``deliver``
(Telegram sends that report success), ``tools``, ``claims`` and ``render``
(the reply), ``models`` (the chat model behind a common interface),
``profiles`` (prompts and tools per mode) and ``agent`` (one turn).
"""
