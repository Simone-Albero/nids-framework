import logging
import functools


class TransformationBuilder:

    __slots__ = [
        "_transformations",
    ]

    def __init__(self) -> None:
        self._transformations: list[callable] = []

    def add_step(self, order: int) -> callable:
        def decorator(func: callable) -> callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper.__name__ = func.__name__
            wrapper.order = order
            self._transformations.append(wrapper)

            logging.info(
                f"Added '{func.__name__}' to transformations pipeline with order {order}."
            )

            return wrapper

        return decorator

    def build(self) -> list[callable]:
        transformations = sorted(
            self._transformations,
            key=lambda transform_function: transform_function.order,
        )
        logging.info(
            f"Generated transformations pipeline with {len(self._transformations)} transformations.\n"
        )

        self._transformations = []
        return transformations
