"""The rail: what staff read, in two jobs (monitor the houses, turn clips into training data) plus admin tools.

Keys stay the screens' internal names (tests, the palette and open_filter use them); the text is what is shown.
"""

NAV = [('MONITOR', [('Fleet', 'Fleet'), ('Review', 'Activity')]),
       ('STUDIO', [('Inbox', 'Inbox'), ('Tag', 'Tag · AI'), ('Label', 'Tag · YOLO'), ('Studio', 'Batches')]),
       ('ADMIN', [('Audit', 'Audit'), ('Index problems', 'Index problems')])]
ACCESS = {'admin': {'Fleet', 'Review', 'Inbox', 'Tag', 'Label', 'Studio', 'Audit', 'Index problems'},
          'support': {'Fleet', 'Review', 'Studio'},
          'labeler': {'Review', 'Label', 'Studio'}}
NAV_TEXT = {name: text for _, items in NAV for name, text in items}
SCREEN_OF = {text: name for name, text in NAV_TEXT.items()}


def screens_for(role):
    """The internal names of the screens *role* may open, in rail order."""
    return [name for _, items in NAV for name, _ in items if name in ACCESS[role]]
