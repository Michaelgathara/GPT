
def get_llama_nemotron_data():
    from datasets import load_dataset, DatasetDict
    dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1")
    
    filtered_dataset = DatasetDict({
        split: dataset[split] for split in ['code', 'science'] if split in dataset
    })
    
    for split in filtered_dataset:
        print(f"  {split}: {len(filtered_dataset[split])} examples")
    
    return filtered_dataset

"""
    Example object view, the numbers are made up
    DatasetDict({
        code: Dataset({
            features: ['input', 'output', 'category', 'license', 'reasoning', 'generator'],
            num_rows: 2801350
        })
        science: Dataset({
            features: ['input', 'output', 'category', 'license', 'reasoning', 'generator'],
            num_rows: 1801350
        })
        math: Dataset({
            features: ['input', 'output', 'category', 'license', 'reasoning', 'generator'],
            num_rows: 465621
        })
        chat: Dataset({
            features: ['input', 'output', 'category', 'license', 'reasoning', 'generator'],
            num_rows: 3760
        })
    })
"""