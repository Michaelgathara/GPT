from datasets import load_dataset, IterableDatasetDict, DatasetDict, IterableDataset
from typing import Union

def get_fineweb_data(
    name: str = "default",
    streaming: bool = True,
    split: str = "train"
) -> Union[IterableDataset, DatasetDict, IterableDatasetDict]:
    print(f"Loading FineWeb-Edu dataset: '{name}' config, split: '{split}', streaming: {streaming}")
    try:
        fineweb_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=name,
            split=split,
            streaming=streaming,
            # trust_remote_code=True # Might be needed depending on HF version/dataset script
        )
        print("Dataset loaded successfully.")
        return fineweb_dataset
    except Exception as e:
        print(f"Error loading dataset HuggingFaceFW/fineweb-edu: {e}")
        print("Please ensure you have 'datasets' installed and are logged into Hugging Face CLI if necessary.")
        raise

# Example Usage 
# if __name__ == '__main__':
#     # Load the full dataset in streaming mode (recommended)
#     streamed_dataset = get_fineweb_data(name="default", streaming=True)
#     print("Streamed Dataset Info:", streamed_dataset)
#     # Iterate over the first few examples
#     count = 0
#     for example in streamed_dataset:
#         print(example['text'][:200]) # Print first 200 chars
#         count += 1
#         if count >= 5:
#             break
#
#     print("-" * 20)
#
#     # Load the 10B token sample without streaming (might fit in memory/cache)
#     # Note: Still large, streaming might still be preferred depending on RAM
#     # sample_dataset = get_fineweb_data(name="sample-10BT", streaming=False)
#     # print("Sample Dataset Info:", sample_dataset)
#     # print(sample_dataset[0]['text'][:200])