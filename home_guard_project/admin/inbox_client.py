"""Backend calls of the Inbox (cloud routes /v1/inbox), for the HTTP and the demo backend."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .backend import ValidationError
from .models import InboxItem, decode


class HttpInbox:
    def inbox(self, **filters):
        """filters: from_utc, to_utc (ISO), customer_id, camera, owner_label, handled ('all' / 'handled' /
        'unhandled'), before_id, limit."""
        params = {k: v for k, v in filters.items() if v is not None}
        return decode(list[InboxItem], self._tag_json('GET', 'inbox', params=params))

    def inbox_decide(self, feedback_id, decision, note=''):
        return decode(InboxItem, self._tag_json('POST', f'inbox/{int(feedback_id)}/decision',
                                                json=dict(decision=decision, note=note)))

    def inbox_reopen(self, feedback_id):
        return decode(InboxItem, self._tag_json('DELETE', f'inbox/{int(feedback_id)}/decision'))


class DemoInbox:
    """The bundled demo answers (demo_data/inbox.json), decisions kept in memory."""

    def _inbox_items(self):
        if getattr(self, '_inbox', None) is None:
            raw = json.loads((Path(__file__).parent / 'demo_data' / 'inbox.json').read_text(encoding='utf-8'))
            self._inbox = decode(list[InboxItem], raw)
        return self._inbox

    def inbox(self, from_utc=None, to_utc=None, customer_id=None, camera=None, owner_label=None, handled='unhandled',
              before_id=None, limit=200):
        start = datetime.fromisoformat(from_utc) if from_utc else None
        end = datetime.fromisoformat(to_utc) if to_utc else None
        def keep(i):
            when = i.received_utc
            return ((start is None or (when is not None and when >= start)) and
                    (end is None or (when is not None and when < end)) and
                    (customer_id is None or i.customer_id == customer_id) and (not camera or i.camera == camera) and
                    (owner_label is None or i.owner_label == owner_label) and
                    (handled == 'all' or (i.decision is not None) == (handled == 'handled')) and
                    (before_id is None or i.feedback_id < before_id))
        items = sorted((i for i in self._inbox_items() if keep(i)), key=lambda i: -i.feedback_id)
        return deepcopy(items[:limit])

    def _inbox_one(self, feedback_id):
        item = next((i for i in self._inbox_items() if i.feedback_id == feedback_id), None)
        if item is None:
            raise ValidationError('Answer not found')
        return item

    def inbox_decide(self, feedback_id, decision, note=''):
        item = self._inbox_one(feedback_id)
        if decision == 'accepted' and not item.consent_training:
            raise ValidationError(f'{item.customer} has withdrawn consent to training use: this answer cannot '
                                  'become a training label')
        item.decision, item.decision_note = decision, note
        item.decided_by, item.decided_utc = self.me().name, datetime.now(timezone.utc)
        return deepcopy(item)

    def inbox_reopen(self, feedback_id):
        item = self._inbox_one(feedback_id)
        item.decision = item.decided_by = item.decided_utc = None
        item.decision_note = ''
        return deepcopy(item)
