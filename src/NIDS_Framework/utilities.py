import os
import pickle

class CacheUtils(object):
    @staticmethod
    def exists(cache_path: str):
        """Check if cache file exists."""
        return os.path.exists(cache_path)
        
    @staticmethod
    def read(cache_path: str):
        """Load data from cache."""
        print(f"Reading directly from cache '{cache_path}' ...")
        try:
            with open(cache_path, 'rb') as r:
                return pickle.load(r)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {cache_path}") from e
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError("Error unpickling data.") from e

    @staticmethod
    def write(cache_path: str, data):
        """Save data to cache."""
        print(f"Writing to cache '{cache_path}' ...")
        try:
            with open(cache_path, 'wb') as file:
                pickle.dump(data, file)
        except Exception as e:
            print(f"Error writing to file: {e}")