import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load better model (FLAN T5 Large)
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Chat function
def chat(user_input):
    prompt = f"Answer this: {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio UI
gr.Interface(fn=chat, inputs="text", outputs="text", title="FLAN T5 Large Chatbot").launch(share=True)
