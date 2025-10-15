import os, logging

try:
    from icecream import ic
except ImportError:
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)

_debug = bool(int(os.getenv("MBO_DEBUG", "0")))
_level = logging.DEBUG if _debug else logging.INFO

_root = logging.getLogger("mbo")
_root.setLevel(_level)
_root.propagate = False

_h = logging.StreamHandler()
_root.addHandler(_h)

_extra_handlers: list[logging.Handler] = []


def attach(handler: logging.Handler):
    """Attach a global handler (e.g. GUI log viewer)."""
    _extra_handlers.append(handler)
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name.startswith("mbo."):
            logger.addHandler(handler)


def get(subname: str | None = None) -> logging.Logger:
    name = "mbo" if subname is None else f"mbo.{subname}"
    logger = logging.getLogger(name)
    logger.setLevel(_level)
    logger.propagate = False
    for h in _extra_handlers:
        if h not in logger.handlers:
            logger.addHandler(h)
    return logger


def enable(*subs):
    for s in subs:
        get(s).disabled = False


def disable(*subs):
    for s in subs:
        get(s).disabled = True


def get_package_loggers():
    return [
        name for name in logging.Logger.manager.loggerDict
        if name.startswith("mbo.")
           and isinstance(logging.Logger.manager.loggerDict[name], logging.Logger)
    ]
