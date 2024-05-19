import functools
from dataset import Dataset
from typing import Callable, List
import inspect
    
class Pipeline(object):
    __slots__ = ['_tasks']

    def __init__(self) -> None:
        """
        Specifies a sequence of tasks to be executed as a preprocessing pipeline.

        Params:
            - _tasks: list of tasks.
        """
        self._tasks: List[Callable[[Dataset], None]] = []
    
    def register(self, priority: int) -> Callable[[Callable[[Dataset], None]], Callable[[Dataset], None]]:
        """
        A decorator function used to incorporate a function into the preprocessing pipeline with a given priority.
        
        Args:
            - priority (int): Priority level of the task.

        Returns:
            - decorator: A decorator function that registers the wrapped function with the specified priority level.
        """
        def decorator(func: Callable[[Dataset], None]) -> Callable[[Dataset], None]:
            params_count = len(inspect.signature(func).parameters)
            if params_count != 1:
                raise TypeError(f"Pipeline tasks require exactly 1 argument, but {params_count} were given.")
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            wrapper.priority = priority
            self._tasks.append(wrapper)
            return wrapper
        return decorator

    def execute(self, dataset: Dataset) -> None:
        """
        Executes the preprocessing pipeline on the given dataset.

        Args:
            - dataset (Dataset): The dataset to be preprocessed.
        """
        for task in sorted(self._tasks, key=lambda wrapper: wrapper.priority):
            task(dataset)