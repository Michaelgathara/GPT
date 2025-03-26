def get_llama_nemotron_data():
    from datasets import load_dataset
    dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1")
    
    filtered_dataset = dataset.filter(lambda example: example['split'] in ['code', 'science'])
    
    for split in filtered_dataset:
        print(f"  {split}: {len(filtered_dataset[split])} examples")
    
    return filtered_dataset