import os
import logging
import re

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


def get_package_loggers():
    return [
        name
        for name in logging.Logger.manager.loggerDict
        if name.startswith("mbo.")
        and isinstance(logging.Logger.manager.loggerDict[name], logging.Logger)
    ]


def _run_key(line: str) -> str:
    """Similarity key for consecutive log lines: leading text, digits masked.

    Masking digits makes indexed lines ("...blocks.0...", "...blocks.1...")
    compare equal, which is what makes per-tensor error floods collapse.
    """
    return re.sub(r"\d+", "#", line[:48]).strip()


def condense(text: str, keep: int = 3, max_lines: int = 40) -> str:
    """Collapse runs of near-identical lines, then cap the total.

    Bounds error text headed for a status sidecar or the GUI log: a single
    torch ``load_state_dict`` failure is one near-identical line per tensor,
    thousands of them.
    """
    if not text:
        return text

    out: list[str] = []
    key: str | None = None
    count = 0

    def close_run() -> None:
        nonlocal key, count
        if count > keep:
            out.append(f"... {count - keep} more similar lines suppressed")
        key, count = None, 0

    for line in text.splitlines():
        this = _run_key(line)
        if this and this == key:
            count += 1
            if count <= keep:
                out.append(line)
            continue
        close_run()
        key, count = this, 1
        out.append(line)
    close_run()

    if len(out) > max_lines:
        hidden = len(out) - max_lines
        out = out[:max_lines] + [f"... {hidden} more lines truncated"]
    return "\n".join(out)
