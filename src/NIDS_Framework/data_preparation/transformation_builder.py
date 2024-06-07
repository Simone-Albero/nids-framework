from typing import Callable, List
import logging


class TransformationBuilder:

    __slots__ = [
        "_transformations",
    ]

    def __init__(self) -> None:
        self._transformations: List[Callable] = []
    
    def add_step(self, order: int) -> Callable:
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            wrapper.order = order
            self._transformations.append(wrapper)
            logging.info(
                f"Added '{func.__name__}' to transformations pipeline with order {order}."
            )
            return wrapper
        return decorator

    def build(self) -> List[Callable]:
        logging.info(
            f"Constructing transformations pipeline with {len(self._transformations)} transformations..."
        )
        transformations = sorted(
            self._transformations,
            key=lambda transform_function: transform_function.order,
        )
        self._transformations = []
        return transformations
