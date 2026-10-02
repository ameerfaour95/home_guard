"""Facts and homeowner guidance, separate from the sanitized support log."""
from dataclasses import dataclass
import re
from .strings import tr

def network_facts(line):
    # Retain only explicitly identified box/network facts, never camera addresses.
    facts = {}
    if re.search(r'box|wi-fi|wifi|ssid|ipv4|ip address|local_ip', line, re.I):
        address = re.search(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])', line)
        if address: facts['address'] = address[0]
    name = re.search(r'(?:wi-fi|wifi|ssid)[\"\']?\s*(?:[:=]|is|on|to)?\s*[\"\']([^\"\']+)', line, re.I)
    if name: facts['network'] = name[1]
    return facts

@dataclass
class FailureFacts:
    network: str = ''
    address: str = ''
    refused: int = 0
    total: int = 0
    no_devices: bool = False
    message: str = ''

    def feed(self, event):
        facts = getattr(event, 'facts', {})
        self.network = facts.get('network', self.network)
        self.address = facts.get('address', self.address)
        count = re.search(r'(\d+) of (\d+) (?:device\(s\)|devices|cameras).*refused', event.text, re.I)
        if count: self.refused, self.total = map(int, count.groups())
        if 'no device answers' in event.text.lower(): self.no_devices = True
        if event.kind == 'step' and event.status == 'fail': self.message = event.text

    def content(self, step):
        if step == 'cameras':
            if self.refused:
                return tr('failure_login_title', count=self.refused, total=self.total), tr('camera_login_help')
            location = tr('failure_location', network=self.network, address=self.address) if self.network and self.address else tr('failure_location_unknown')
            return tr('failure_cameras_title'), tr('failure_cameras_body', location=location)
        if step == 'network':
            return tr('failure_network_title', network=self.network or tr('failure_home_network')), tr('failure_network_body')
        if step == 'update': return tr('failure_update_title'), tr('failure_update_body')
        if step == 'connect': return tr('failure_connect_title'), tr('failure_connect_body')
        return tr('failure_other_title'), self.message or tr('failure_other_body')
