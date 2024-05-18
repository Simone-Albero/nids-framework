import functools

class Pipeline:
    def __init__(self):
        self._tasks = []
    
    def register(self, priority):
        """
        A decorator function used to incorporate a function into the preprocessing pipeline with a given priority.
        
        Args:
            - priority (int): Priority level of the task.

        Returns:
            - decorator: A decorator function that registers the wrapped function with the specified priority level.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            wrapper.priority = priority
            self._tasks.append(wrapper)
            return wrapper
        return decorator

    def execute(self):
        for task in sorted(self._tasks, key=lambda x: x.priority):
            task()