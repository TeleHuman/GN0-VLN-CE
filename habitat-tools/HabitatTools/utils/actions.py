ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

VALID_ACTIONS = {ACTION_STOP, ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT}


def clamp_action(action: int, fallback: int = ACTION_STOP) -> int:
    try:
        value = int(action)
    except (TypeError, ValueError):
        return int(fallback)
    return value if value in VALID_ACTIONS else int(fallback)


def choose_first_action(action_seq, fallback: int = ACTION_STOP) -> int:
    if not action_seq:
        return clamp_action(fallback, ACTION_STOP)
    return clamp_action(action_seq[0], fallback)
