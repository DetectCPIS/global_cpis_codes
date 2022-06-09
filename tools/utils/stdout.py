import sys

stdout = sys.stdout


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


def stdout_on():
    sys.stdout = stdout


def stdout_off():
    if not is_debug():
        sys.stdout = None
