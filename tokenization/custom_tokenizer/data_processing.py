from datasets import Dataset
from .config import DATA_PATH, NUM_CORES
from data import get_wikitext_data, clean_textdata

def load_dataset() -> Dataset:
    dataset = get_wikitext_data()
    print("Total lines:", len(dataset["train"]["text"]))
    return dataset["train"]

def clean_batch(examples):
    """Cleans text data in batches."""
    cleaned_texts = [clean_textdata(text) for text in examples["text"]]
    cleaned_texts = list(filter(None, cleaned_texts))
    return {"text": cleaned_texts}

def clean_and_save_dataset(dataset: Dataset):
    """Cleans and saves dataset to a file."""
    cleaned_dataset = dataset.map(
        clean_batch,
        batched=True,
        batch_size=10_000,
        num_proc=NUM_CORES,
        desc="Cleaning text"
    )

    print("Cleaned lines:", len(cleaned_dataset["text"]))

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_dataset["text"]) + "\n")

    return DATA_PATH