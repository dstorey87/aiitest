import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize logging with a dedicated log file
logging.basicConfig(
    filename='/app/logs/inference.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set the new model name
model_name = "huggingface/your-new-model-name"  # Replace with the actual model name

# Set explicit cache directory path to use previously downloaded shards
cache_dir = "/root/.cache/huggingface"

def load_model():
    try:
        logging.info('Starting to load Falcon model...')
        # Load tokenizer and model from the pre-trained cache
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map='auto',
            torch_dtype="auto",
        )
        logging.info('Model loaded successfully.')
        return tokenizer, model
    except Exception as e:
        logging.error(f'Failed to load model: {e}', exc_info=True)
        raise e

def generate_text(tokenizer, model, prompt, max_length=500, temperature=0.75):
    try:
        logging.info(f'Generating text from the model with prompt: {prompt}')
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        output = model.generate(
            **inputs,
            max_length=max_length,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,  # Set to True for more variability in text generation
            early_stopping=True
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logging.info('Text generation successful.')
        return generated_text
    except Exception as e:
        logging.error(f'Error generating text: {e}', exc_info=True)
        raise e

def main():
    try:
        # Load model once at the beginning
        tokenizer, model = load_model()
        
        # Define a specific and structured prompt
        prompt = """
        Write a comprehensive article discussing:
        1. The definition of renewable energy.
        2. The main benefits of renewable energy sources such as solar, wind, and hydroelectric power.
        3. How renewable energy helps to combat climate change.
        4. The economic benefits and job creation through renewable energy.
        5. Conclusion on the future impact of renewable energy globally.
        """
        
        # Generate text using the Falcon model
        generated_text = generate_text(tokenizer, model, prompt, max_length=750, temperature=0.7)
        print("Generated Text:\n")
        print(generated_text)
        
        # Save generated text to a file
        with open('generated_text.txt', 'w', encoding='utf-8') as f:
            f.write(generated_text)
        logging.info('Generated text saved to generated_text.txt')

        # Terminate after successful generation
        return

    except Exception as e:
        logging.error(f'An error occurred in main(): {e}', exc_info=True)
        print("An error occurred. Check the log file for details.")

if __name__ == '__main__':
    main()
