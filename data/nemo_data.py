def get_nemo_data():
    from datasets import load_dataset
    nemo_dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1")
    return nemo_dataset