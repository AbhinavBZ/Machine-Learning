
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Available translation models
MODELS = {
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en"
}

# Select CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store loaded models to avoid loading them repeatedly
model_cache = {}


def load_model(direction):

    if direction not in model_cache:

        model_name = MODELS[direction]

        print("\nLoading translation model...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)

        model.to(device)
        model.eval()

        model_cache[direction] = (tokenizer, model)

        print("Model loaded successfully!")

    return model_cache[direction]


def translate(text, direction):

    tokenizer, model = load_model(direction)

    # Convert text into model input tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    # Move input tensors to CPU/GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate translation
    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    # Convert generated tokens into readable text
    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result


def main():

    print("=" * 45)
    print("       AI TEXT TRANSLATION SYSTEM")
    print("=" * 45)

    print("1. English to Hindi")
    print("2. Hindi to English")

    choice = input("\nSelect translation direction: ").strip()

    if choice == "1":
        direction = "en-hi"
    elif choice == "2":
        direction = "hi-en"
    else:
        print("Invalid selection.")
        return

    print("\nEnter 'exit' to close the program.")

    while True:

        text = input("\nEnter text to translate: ").strip()

        if text.lower() == "exit":
            print("Translation system closed.")
            break

        if not text:
            print("Please enter some text.")
            continue

        try:
            result = translate(text, direction)

            print("\nTranslated Output:")
            print(result)

        except Exception as error:
            print("\nTranslation error:", error)


if __name__ == "__main__":
    print("Device:", device)
    main()
