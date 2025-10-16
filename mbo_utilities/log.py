import os, logging

try:
    from icecream import ic
except ImportError:
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)

_level_override: int | None = None

def set_global_level(level: int):
    global _level_override
    _level_override = level
    _root.setLevel(level)
    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and name.startswith("mbo"):
            lg.setLevel(level)

def get(subname: str | None = None) -> logging.Logger:
    name = "mbo" if subname is None else f"mbo.{subname}"
    lg = logging.getLogger(name)
    eff = _level_override if _level_override is not None else _level
    if lg.level in (logging.NOTSET,):  # only set if not customized
        lg.setLevel(eff)
    lg.propagate = False
    for h in _extra_handlers:
        if h not in lg.handlers:
            lg.addHandler(h)
    return lg

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
