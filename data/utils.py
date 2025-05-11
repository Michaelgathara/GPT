from datasets import load_from_disk
def save_data(dataset, path):
    dataset.save_to_disk(path)

def load_data(path):
    return load_from_disk(path)