"""Local Tailscale inventory; never opens a connection to a peer."""
from dataclasses import dataclass
import json
import os
import subprocess
from .remote_cameras import target_user

@dataclass(frozen=True)
class Peer:
    name: str
    address: str
    online: bool

def parse_peers(output):
    try:
        data=json.loads(output)
        peers=data.get('Peer', {})
        values=peers.values() if isinstance(peers,dict) else peers
        result={}
        for peer in values:
            if str(peer.get('OS','')).lower() != 'windows': continue
            addresses=peer.get('TailscaleIPs', [])
            address=next((a for a in addresses if ':' not in a), addresses[0] if addresses else '')
            if not address: continue
            target_user('user@'+address)
            name=str(peer.get('HostName') or peer.get('DNSName') or address).rstrip('.')
            result[address]=Peer(name,address,bool(peer.get('Online',False)))
        return sorted(result.values(), key=lambda p:(not p.online,p.name.lower(),p.address))
    except (ValueError,TypeError,AttributeError): return []

def discover(runner=subprocess.run):
    env=os.environ.copy();env.pop('VIRTUAL_ENV',None)
    try:
        result=runner(['tailscale','status','--json'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return parse_peers(result.stdout) if result.returncode == 0 else []
    except (OSError,subprocess.TimeoutExpired): return []

def compose_target(user,address):
    value=user.strip()+'@'+address.strip()
    target_user(value)
    return value
