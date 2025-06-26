from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Choose model
model_name = "facebook/blenderbot-400M-distill"

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize conversation history
conversation_history = []

# Chat loop
while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get user input
    input_text = input("> ")

    # Tokenize input and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate model response
    outputs = model.generate(**inputs)

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Print bot reply
    print(f"Bot: {response}")

    # Update history
    conversation_history.append(input_text)
    conversation_history.append(response)
