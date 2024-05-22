import pickle

def file_read(file_path: str) -> None:
    """Read data directly from the file.
    
    Args:
        file_path: The path of the file to read from.
    
    Raises:
        FileNotFoundError: Raised if the file does not exist.
    """
    print(f"Reading directly from file: {file_path} ...")
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError("Error unpickling data.") from e

def file_write(file_path: str, data) -> None:
    """Write the data directly to the file.
    
    Args:
        file_path: The destination file path.
        data: The data to be stored.    
    """
    print(f"Writing to file: {file_path} ...")
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error writing to file: {e}")