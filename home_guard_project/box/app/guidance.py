"""Field guidance and failed-step ownership, independent of Qt."""
import re

def field_errors(page, values, wifi=False, find=True):
    errors = {}
    if page == 0 and not re.fullmatch(r"[^@\s]+@[^@\s]+",values.get("address", "").strip()):
        errors["address"] = "address_error"
    if page == 1 and wifi:
        if not values.get("ssid", "").strip(): errors["ssid"] = "ssid_error"
        if not 8 <= len(values.get("wifi_password", "")) <= 63: errors["wifi_password"] = "wifi_password_error"
    if page == 2 and not re.fullmatch(r"[a-z0-9_]+",values.get("house", "").strip()):
        errors["house"] = "house_error"
    if page == 3 and find:
        for key in ("camera_user", "camera_password"):
            if not values.get(key, "").strip(): errors[key] = key+"_error"
    return errors

def retry_page(step):
    return {"update":0,"name_step":2,"network_step":1,"camera_step":3,"readiness":0}[step]
