import random

# Cookie getter and setter functions
def get_cookie(cookie_manager, name):
    return cookie_manager.get(name)


def set_cookie(cookie_manager, name, value):
    cookie_manager.set(name, value)


# Randomly assign user to a variant (50% chance for A or B)
def get_variant(cookie_manager):
    if get_cookie(cookie_manager, "variant"):
        return get_cookie(cookie_manager, "variant")
    else:
        variant = random.choice(["A", "B"])
        set_cookie(cookie_manager, "variant", variant)
        return get_cookie(cookie_manager, "variant")