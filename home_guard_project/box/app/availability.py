"""Presentation decisions for a local box or an offline installer computer."""
def stage_state(*, reachable, cameras, pictures=True):
    if not reachable: return 'box_unreachable'
    if not cameras: return 'no_cameras'
    if not pictures: return 'pictures_off'
    return None

def network_description(answers):
    from .strings import tr
    return tr('wifi') + ' · ' + answers.ssid if answers.network == 'wifi' else tr('ethernet')
