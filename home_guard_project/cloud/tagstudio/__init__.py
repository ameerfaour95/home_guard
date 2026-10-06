"""The tagging studio: category tags for the AI over every clip we have.

Sources (``sources.py``): the unified dataset's old tags, the customers' alerts and answers (the indexed events, and
the local ``owner_feedback/`` copies), and a teacher's suggestion (``teacher.py``: the eval results now, a self-hosted
model later). The work queue (``queue.py``) puts contradictions first. Tags are append-only ``tag_events``
(``fields.py``). Exports (``export.py``) write the VLM training JSONL and eval manifest rows locally.
Categories come from ``fleet_contract/taxonomy.py`` (a copy of ``box/taxonomy.py``).
"""
