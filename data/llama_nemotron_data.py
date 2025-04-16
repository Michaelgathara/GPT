
def get_llama_nemotron_data():
    from datasets import load_dataset, DatasetDict
    dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1", split=['code', 'science'])
    
    return dataset

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
    })
"""