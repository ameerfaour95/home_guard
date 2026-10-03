"""Alert choices shared by settings, camera transports and live confirmation."""
import json
import re
from ..boxconfig import SET_OPTIONS
from .sensitivity import CameraSensitivityBackend, parse_thresholds

TYPES = SET_OPTIONS['alert_on']


def ordered_types(value):
    if isinstance(value, str):
        value = value.split(',')
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
        raise ValueError('Choose People, Vehicles or Animals.')
    picked = {v.strip().lower() for v in value if v.strip()}
    if not picked or not picked <= set(TYPES):
        raise ValueError('Choose People, Vehicles or Animals.')
    return tuple(v for v in TYPES if v in picked)


def camera_alert_operation(name, value):
    if not re.fullmatch(r'[a-z0-9_]+', name):
        raise ValueError('Invalid camera name')
    return ['set-camera-alerts', '--camera', name, *(['--default'] if value is None else ['--on', ','.join(ordered_types(value))])]


def alert_result(output):
    # Decoder diagnostics can precede the JSON result, including multiline JSON.
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != '{':
            continue
        try:
            data, _ = decoder.raw_decode(output[index:])
        except ValueError:
            continue
        if isinstance(data, dict):
            if 'error' in data:
                raise ValueError(str(data['error']))
            return data
    raise ValueError('The camera alert choices could not be read. Try again.')


class CameraAlertsBackend(CameraSensitivityBackend):
    def set_house_alerts(self, value):
        from dataclasses import replace
        from .box_controls import RemoteSettingsBackend, Settings
        value = ','.join(ordered_types(value))
        if getattr(self, 'remote', False) and not self.box.demo:
            backend = RemoteSettingsBackend(self.target, Settings(alert_on=','.join(self.camera_alerts()['house'])), self.runner, self.alert_reported_status, key=self.key)
        else:
            backend = self.box
        backend.save_settings(replace(backend.load_settings(), alert_on=value))
        return list(ordered_types(value))

    def alert_reported_status(self):
        if not getattr(self, 'remote', False) or self.box.demo:
            return self.box.reported_status()
        command = r'cd /d C:\home_guard && type logs\ai_status.json'
        result = self.command(['ssh.exe', '-i', str(self.key), '-o', 'LogLevel=ERROR', self.target, command], allow_failed=True)
        return alert_result(result.stdout) if not result.returncode else {}

    def _alert_command(self, args):
        if getattr(self, 'remote', False):
            result = self.command(self.ssh(' '.join(args)), allow_failed=True)
            data = alert_result(result.stdout)
            code = result.returncode
        else:
            code, data = self.command(*args)
        if code:
            raise ValueError(data.get('error', 'The camera alert choices could not be saved. Try again.'))
        return data

    def camera_alerts(self):
        if self.box.demo:
            own = getattr(self.box, 'camera_alert_on', {})
            return {'house': list(ordered_types(self.box.load_settings().alert_on)),
                    'house_sensitivity': self.box.load_settings().effective_sensitivity(),
                    'cameras': [{'name': c.name, 'alert_on': list(own[c.name]) if c.name in own else None,
                                 'sensitivity': dict(self.box.camera_sensitivity[c.name]) if c.name in self.box.camera_sensitivity else None} for c in self.records]}
        data = self._alert_command(['camera-alerts'])
        return {'house': list(ordered_types(data['house'])),
                'house_sensitivity': parse_thresholds(data['house_sensitivity']) if data.get('house_sensitivity') else None,
                'cameras': [{'name': row['name'], 'alert_on': None if row['alert_on'] is None else list(ordered_types(row['alert_on'])),
                             'sensitivity': parse_thresholds(row['sensitivity']) if row.get('sensitivity') else None} for row in data['cameras']]}

    def set_camera_alerts(self, name, value):
        args = camera_alert_operation(name, value)
        if self.box.demo:
            if name not in {c.name for c in self.records}:
                raise ValueError('This camera is no longer available.')
            if value is None:
                self.box.camera_alert_on.pop(name, None)
            else:
                self.box.camera_alert_on[name] = list(ordered_types(value))
            return self.box.camera_alert_on.get(name)
        data = self._alert_command(args)
        if data.get('camera') != name or 'alert_on' not in data:
            raise ValueError('The camera alert choice could not be confirmed. Try again.')
        return None if data['alert_on'] is None else list(ordered_types(data['alert_on']))
