from datasets import load_dataset
def get_fineweb_data(type: int):
    if type == 0:
        fineweb_dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default")
    if type == 1:
        fineweb_dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")
    return fineweb_dataset