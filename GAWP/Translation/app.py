
from flask import Flask, render_template, request
import torch
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Translation models
MODELS = {
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en"
}

# Cache loaded models
model_cache = {}

# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model(direction):

    if direction not in MODELS:
        raise ValueError("Unsupported translation direction")

    if direction not in model_cache:

        model_name = MODELS[direction]

        print(f"\nLoading model: {model_name}")

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        model.to(device)
        model.eval()

        model_cache[direction] = (tokenizer, model)

        print(f"Model loaded successfully on {device}")

    return model_cache[direction]


def translate(text, direction):

    tokenizer, model = load_model(direction)

    # Convert input text into tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Move tokens to selected device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate translated tokens
    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    # Convert tokens into readable text
    result = tokenizer.decode(
        translated_tokens[0],
        skip_special_tokens=True
    )

    return result


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    original = ""

    if request.method == "POST":

        original = request.form.get("text", "").strip()
        source = request.form.get("source", "en")
        target = request.form.get("target", "hi")

        direction = f"{source}-{target}"

        if not original:
            result = "Please enter some text."

        elif source == target:
            result = "Please select different source and target languages."

        elif direction not in MODELS:
            result = "Selected translation direction is not supported."

        else:
            try:
                result = translate(original, direction)

            except Exception as error:
                print("Translation error:", error)
                result = "Translation failed. Please try again."

    return render_template(
        "index.html",
        result=result,
        original=original
    )


if __name__ == "__main__":

    print("Starting AI Translation Application...")
    print("Device:", device)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
