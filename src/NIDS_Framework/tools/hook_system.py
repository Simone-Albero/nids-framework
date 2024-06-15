from typing import Dict, Callable


class HookSystem:

    __slots__ = [
        "_hooks",
    ]

    def __init__(self) -> None:
        self._hooks: Dict[str, Callable] = {}

    def register_hook(self, event: str, func: Callable) -> None:
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(func)

    def execute_hooks(self, event: str, *args, **kwargs) -> None:
        if event in self._hooks:
            for func in self._hooks[event]:
                func(*args, **kwargs)