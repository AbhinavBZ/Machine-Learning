
from flask import Flask, render_template, request
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
import torch

app = Flask(__name__)

MODEL_NAME = "deepset/roberta-base-squad2"

# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading Question Answering model...")

# Load tokenizer and pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("Question Answering model loaded successfully!")


def get_answer(context, question):

    # Tokenize question and context together
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=512
    )

    # Identify tokens belonging to the context
    sequence_ids = inputs.sequence_ids(0)

    # Keep offset mapping separately
    offsets = inputs.pop("offset_mapping")[0].tolist()

    # Move model inputs to CPU/GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Find context token positions
    context_positions = [
        i for i, seq_id in enumerate(sequence_ids)
        if seq_id == 1
    ]

    if not context_positions:
        return "No answer found in the provided context.", 0.0

    # Find the best valid answer span
    best_score = float("-inf")
    best_start = None
    best_end = None

    for start in context_positions:
        for end in context_positions:

            if end < start:
                continue

            # Limit answer length
            if end - start + 1 > 30:
                continue

            score = (
                start_logits[start].item()
                + end_logits[end].item()
            )

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    # Check whether the model predicts no answer
    null_score = (
        start_logits[0].item()
        + end_logits[0].item()
    )

    if null_score > best_score:
        return "No suitable answer found in the context.", 0.0

    # Extract answer using character offsets
    char_start = offsets[best_start][0]
    char_end = offsets[best_end][1]

    answer = context[char_start:char_end].strip()

    # Approximate confidence from start/end probabilities
    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)

    confidence = (
        start_probs[best_start].item()
        * end_probs[best_end].item()
        * 100
    )

    return answer, round(confidence, 2)


@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""
    confidence = None
    context = ""
    question = ""

    if request.method == "POST":

        context = request.form.get("context", "").strip()
        question = request.form.get("question", "").strip()

        if not context or not question:
            answer = "Please provide both context and question."

        else:
            try:
                answer, confidence = get_answer(
                    context,
                    question
                )

            except Exception as error:
                print("Error:", error)
                answer = "Unable to process the question."

    return render_template(
        "index.html",
        answer=answer,
        confidence=confidence,
        context=context,
        question=question
    )


if __name__ == "__main__":

    print("Starting Question Answering Application...")
    print("Device:", device)

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
