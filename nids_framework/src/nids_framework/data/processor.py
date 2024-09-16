import logging


class Processor:

    __slots__ = [
        "_transformations",
    ]

    def __init__(self, transformations: list[callable]) -> None:
        self._transformations = transformations

    def apply(self, data: any) -> None:
        logging.info(f"Applying {len(self._transformations)} transformation...")
        for transform_function in self._transformations:
            try:
                data = transform_function(data)
            except Exception as e:
                logging.error(f"Error applying transformation: {e}")
                exit(1)

        logging.info("Completed.\n")
        return data
