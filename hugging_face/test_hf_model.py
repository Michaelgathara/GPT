import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model(model_path, prompt, max_tokens=100, temperature=0.8, top_k=50):
    print(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    print(f"\nPrompt: \"{prompt}\"")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"Generating with: max_new_tokens={max_tokens}, temperature={temperature}, top_k={top_k}")
    
    try:
        with torch.no_grad():
            output_sequences = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print("\nGenerated text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        print("Generation successful!")
    
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Hugging Face model locally")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved HuggingFace model")
    parser.add_argument("--prompt", type=str, default="Hello, I am a language model", help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )