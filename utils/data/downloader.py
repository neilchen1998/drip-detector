from datasets import load_dataset

def download_dataset(path: str):
    '''Download the dataset from Hugging Face

    path: the path of the dataset
    '''

    try:
        print(f"Downloading dataset from: {path}...\n")
        dataset = load_dataset(path)
        return dataset
    except Exception as e:
        print(f"Error: {e}")
        return
