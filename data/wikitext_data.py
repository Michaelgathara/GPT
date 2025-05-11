from datasets import load_dataset
def get_wikitext_data():
    wikitext_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    return wikitext_dataset

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