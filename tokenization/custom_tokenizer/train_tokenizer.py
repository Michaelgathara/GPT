from .trainer import train_and_save_tokenizer
from .data_processing import load_dataset, clean_and_save_dataset

def main():
    train_dataset = load_dataset()
    clean_and_save_dataset(train_dataset)
    train_and_save_tokenizer()

if __name__ == "__main__":
    main()