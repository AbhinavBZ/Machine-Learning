
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "deepset/roberta-base-squad2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print("\nLoading Question Answering model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    return tokenizer, model


def get_answer(context, question, tokenizer, model):

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True
    )

    offsets = inputs.pop("offset_mapping")[0]
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Find the most likely answer span
    start_index = torch.argmax(start_logits).item()
    end_index = torch.argmax(end_logits).item()

    # Check whether the predicted span is valid
    if end_index < start_index or end_index - start_index > 30:
        return "No suitable answer found."

    start_char = offsets[start_index][0].item()
    end_char = offsets[end_index][1].item()

    answer = context[start_char:end_char]

    if not answer.strip():
        return "No suitable answer found."

    return answer


def main():

    print("=" * 50)
    print("       AI QUESTION ANSWERING SYSTEM")
    print("=" * 50)

    tokenizer, model = load_model()

    print("\nEnter 'exit' to close the program.")

    while True:

        context = input("\nEnter the context passage: ").strip()

        if context.lower() == "exit":
            print("Question Answering System Closed.")
            break

        if not context:
            print("Please enter a valid passage.")
            continue

        question = input("\nEnter your question: ").strip()

        if question.lower() == "exit":
            print("Question Answering System Closed.")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        try:
            answer = get_answer(context, question, tokenizer, model)

            print("\nAnswer:")
            print(answer)

        except Exception as error:
            print("\nError:", error)


if __name__ == "__main__":
    print("Device:", device)
    main()
