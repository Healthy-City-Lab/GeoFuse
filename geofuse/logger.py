"""
Coloured console logger for GeoFuse engines.

Usage
-----
    from geofuse.logger import get_logger

    log = get_logger("GVI")
    log("INFO",  "Starting analysis…")
    log("OK",    "Segmentation complete: veg=0.31")
    log("WARN",  "No panoramas found nearby")
    log("ERROR", "Download failed: ConnectError")

Levels
------
    INFO   cyan    — routine progress
    OK     green   — successful result / file written
    WARN   yellow  — skipped step / missing data (non-fatal)
    ERROR  red     — exception or unrecoverable failure
"""

_ANSI = {
    "INFO": "\033[36m",  # cyan
    "OK": "\033[32m",  # green
    "WARN": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}


def get_logger(engine: str):
    """Return a ``log(level, msg)`` callable prefixed with *engine*.

    Parameters
    ----------
    engine : str
        Short engine name shown in the prefix, e.g. ``"GVI"``, ``"NDVI"``,
        ``"FUSION"``.

    Returns
    -------
    callable
        ``log(level: str, msg: str) -> None``
    """
    tag = engine.upper()

    def log(level: str, msg: str) -> None:
        color = _ANSI.get(level, "")
        print(
            f"{color}{_ANSI['BOLD']}[{tag} {level}]{_ANSI['RESET']} {msg}",
            flush=True,
        )

    return log
