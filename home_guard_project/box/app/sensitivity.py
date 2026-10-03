"""Per-type detector values and camera command transport."""
import re
from ..camera_alerts import parse_thresholds


def camera_sensitivity_operation(name, values):
    if not re.fullmatch(r'[a-z0-9_]+', name):
        raise ValueError('Invalid camera name')
    args = ['set-camera-sensitivity', '--camera', name]
    return args + (['--default'] if values is None else
                   ['--values', ','.join(f'{k}={v:g}' for k, v in parse_thresholds(values).items())])


class CameraSensitivityBackend:
    def set_house_sensitivity(self, values):
        from dataclasses import replace
        from .box_controls import RemoteSettingsBackend, Settings
        values = parse_thresholds(values)
        if getattr(self, 'remote', False) and not self.box.demo:
            house = self.camera_alerts()['house_sensitivity']
            if house is None:
                raise ValueError('Sensitivity values are unavailable. Update the box program.')
            backend = RemoteSettingsBackend(self.target, Settings(**{'conf_'+k: v for k, v in house.items()}),
                                            self.runner, self.alert_reported_status, key=self.key)
        else:
            backend = self.box
        after = replace(backend.load_settings(), **{'conf_'+k: v for k, v in values.items()})
        backend.save_settings(after)
        return after.effective_sensitivity()

    def set_camera_sensitivity(self, name, values):
        args = camera_sensitivity_operation(name, values)
        if self.box.demo:
            if name not in {c.name for c in self.records}:
                raise ValueError('This camera is no longer available.')
            if values is None:
                self.box.camera_sensitivity.pop(name, None)
            else:
                self.box.camera_sensitivity[name] = parse_thresholds(values)
            own = self.box.camera_sensitivity.get(name)
            return None if own is None else dict(own)
        data = self._alert_command(args)
        if data.get('camera') != name or 'sensitivity' not in data:
            raise ValueError('The camera sensitivity could not be confirmed. Try again.')
        return None if data['sensitivity'] is None else parse_thresholds(data['sensitivity'])
