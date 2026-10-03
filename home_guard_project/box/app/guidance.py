"""Field guidance and failed-step ownership, independent of Qt."""
import re
from .setup_pages import Page

def field_errors(page, values, wifi=False, find=True):
    errors = {}
    if page == 0:
        from .remote_cameras import target_user
        try: target_user(values.get("address", "").strip())
        except ValueError: errors["address"] = "address_error"
    if page == 1 and wifi:
        if not values.get("ssid", "").strip(): errors["ssid"] = "ssid_error"
        if not 8 <= len(values.get("wifi_password", "")) <= 63: errors["wifi_password"] = "wifi_password_error"
    if page == 2 and not re.fullmatch(r"[a-z0-9_]+",values.get("house", "").strip()):
        errors["house"] = "house_error"
    if page == Page.OWNER:
        for key, limit, required in (('owner_name', 120, True), ('owner_phone', 200, False), ('installer', 120, False)):
            value = values.get(key, '')
            if (required and not value.strip()) or len(value.strip()) > limit or re.search(r'[\x00-\x1f\x7f]', value):
                errors[key] = key + '_error'
    if page == Page.CAMERAS and find:
        for key in ("camera_user", "camera_password"):
            if not values.get(key, "").strip(): errors[key] = key+"_error"
    return errors

def retry_page(step):
    return {"update":Page.ADDRESS,"name_step":Page.HOUSE,"network_step":Page.NETWORK,"camera_step":Page.CAMERAS,"readiness":Page.ADDRESS}[step]
