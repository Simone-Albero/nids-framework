import logging
from typing import Callable, Any, List


class Processor:

    __slots__ = [
        "_transformations",
    ]

    def __init__(self, transformations: List[Callable[[Any], Any]]) -> None:
        self._transformations = transformations

    def apply(self, data: Any) -> Any:
        logging.info(f"Applying {len(self._transformations)} transformation...")
        for transform_function in self._transformations:
            try:
                data = transform_function(data)
            except Exception as e:
                logging.error(f"Error applying transformation: {e}")
                exit(1)

        logging.info("Completed.\n")
        return data
