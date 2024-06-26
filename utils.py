import pickle as pkl


def load_pkl(file_name: str):
    """
    Load pkl file, if file not found will return None type
    """
    try:
        with open(file_name, 'rb') as file:
            data = pkl.load(file)
        return data
    except FileNotFoundError:
        return None
    
    
    
def save_pkl(data, file_name: str):
    with open(file_name, 'wb') as file:
        pkl.dump(data, file)