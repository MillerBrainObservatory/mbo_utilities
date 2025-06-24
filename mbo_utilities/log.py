import os, logging

# this ic import convenciton is from their readme
try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
    install = None

_debug = bool(int(os.getenv("MBO_DEBUG", "0")))
_level = logging.DEBUG if _debug else logging.INFO

_h = logging.StreamHandler()
_root = logging.getLogger("mbo")
_root.setLevel(_level)
_root.addHandler(_h)
_root.propagate = False


def get(subname: str | None = None) -> logging.Logger:
    name = "mbo" if subname is None else f"mbo.{subname}"
    return logging.getLogger(name)


def enable(*subs):
    for s in subs:
        get(s).disabled = False


def disable(*subs):
    for s in subs:
        get(s).disabled = True


for sub in os.getenv("MBO_ENABLE", "").split(","):
    if sub:
        enable(sub)
for sub in os.getenv("MBO_DISABLE", "").split(","):
    if sub:
        disable(sub)


def get_package_loggers() -> list[str]:
    """Get all loggers that are part of the 'mbo' package."""
    return [
        name
        for name in logging.Logger.manager.loggerDict
        if name.startswith("mbo.")
        and isinstance(logging.Logger.manager.loggerDict[name], logging.Logger)
    ]


def get_all_loggers() -> list[str]:
    """Get all loggers that are currently enabled."""
    return [
        name
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger) and not logger.disabled
    ]
