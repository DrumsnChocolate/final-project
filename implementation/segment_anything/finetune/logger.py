from typing import Callable

from prodict import Prodict


class Logger(Prodict):
    log: Callable[[str], None]