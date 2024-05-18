import functools
from typing import Callable, List

class Pipeline(object):
    __slots__ = ['_tasks']

    def __init__(self) -> None:
        """
        Specifies a sequence of tasks to be executed as a preprocessing pipeline.

        Params:
            - _tasks: list of tasks.
        """
        self._tasks: List[Callable] = []
    
    def register(self, priority: int) -> Callable[[Callable], Callable]:
        """
        A decorator function used to incorporate a function into the preprocessing pipeline with a given priority.
        
        Args:
            - priority (int): Priority level of the task.

        Returns:
            - decorator: A decorator function that registers the wrapped function with the specified priority level.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            wrapper.priority = priority
            self._tasks.append(wrapper)
            return wrapper
        return decorator

    def execute(self) -> None:
        for task in sorted(self._tasks, key=lambda wrapper: wrapper.priority):
            task()