from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import gradio as gr

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

gr.Interface(fn=chat, inputs="text", outputs="text", title="FLAN T5 Chatbot").launch()
