from typing import Callable, List
import logging


class TransformationBuilder:

    __slots__ = [
        "_transformations",
    ]

    def __init__(self) -> None:
        self._transformations: List[Callable] = []

    def add_step(self, order: int) -> Callable:
        def decorator(transform_function: Callable) -> Callable:
            def transform_sample(sample) -> Callable:
                return transform_function(sample)

            transform_sample.order = order
            self._transformations.append(transform_sample)
            logging.info(
                f"Added '{transform_function.__name__}' to transformations pipeline."
            )
            return transform_sample

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
