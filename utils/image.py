import base64
import binascii


def is_base64(value):
    try:
        base64.b64decode(value, validate=True)
        return True
    except binascii.Error:
        return False
    except Exception as e:
        return False