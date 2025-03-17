def get_wikitext_data():
    from datasets import load_dataset
    wikitext_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    return wikitext_dataset

# Function to save data so that tokenization can be saved
def save_data(dataset, path):
    dataset.save_to_disk(path)

# Function to reload the data from local storage
def load_data(path):
    from datasets import load_from_disk
    return load_from_disk(path)

"""
    DatasetDict({
    test: Dataset({
        features: ['text'],
        num_rows: 4358
    })
    train: Dataset({
        features: ['text'],
        num_rows: 1801350
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 3760
    })
})
"""