import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to chat
def chat(user_input, history=[]):
    history_string = "\n".join(history)
    inputs = tokenizer.encode_plus(history_string, user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    history.append(user_input)
    history.append(response)

    # Limit history
    history = history[-6:]

    return response, history

# Gradio Interface
chatbot = gr.Interface(fn=chat,
                       inputs="text",
                       outputs="text",
                       title="Blenderbot Chatbot")

# Launch
chatbot.launch()
