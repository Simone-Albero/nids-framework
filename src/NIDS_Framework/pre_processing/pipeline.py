from typing import Callable, List
import functools
import inspect

from pre_processing import dataset


class Pipeline(object):
    """Specify a sequence of tasks to be executed as a pipeline.

    Attributes:
        _tasks: A list of functions wrapped with a specific priority level
    """

    __slots__ = ["_tasks"]

    def __init__(self) -> None:
        self._tasks: List[Callable[[dataset.Dataset], None]] = []

    def register(
        self, priority: int
    ) -> Callable[
        [Callable[[dataset.Dataset], None]], Callable[[dataset.Dataset], None]
    ]:
        """Wrap a list of functions with a specific priority level and add the function to the task list.

        Args:
            priority: The priority level of the function.

        Raises:
            ValueError: Raised if the number of function parameters is incorrect.
        """

        def decorator(
            func: Callable[[dataset.Dataset], None]
        ) -> Callable[[dataset.Dataset], None]:
            params_count = len(inspect.signature(func).parameters)
            if params_count != 1:
                raise ValueError(
                    f"Pipeline tasks require exactly 1 argument, but {params_count} were given."
                )

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper.priority = priority
            self._tasks.append(wrapper)
            return wrapper

        return decorator

    def execute(self, dataset: dataset.Dataset) -> None:
        """Execute the pipeline on the given dataset.

        Args:
            dataset: The dataset on which the tasks must be executed
        """
        for task in sorted(self._tasks, key=lambda wrapper: wrapper.priority):
            task(dataset)
