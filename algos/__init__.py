from . import fb

__all__ = ["fb", "choose_algo"]


def choose_algo(name: str):
    if name == "fb":
        return fb
    raise ValueError(f"Algorithm {name} is not supported now.")
