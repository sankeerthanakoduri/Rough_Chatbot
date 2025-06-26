import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# App title
st.title("🗨️ Blenderbot Chatbot")

# Conversation history (Streamlit session state)
if "history" not in st.session_state:
    st.session_state["history"] = []

# User input
user_input = st.text_input("You:", "")

# If user clicks "Send"
if st.button("Send"):
    # Combine history
    history_string = "\n".join(st.session_state["history"])

    # Tokenize
    inputs = tokenizer.encode_plus(history_string, user_input, return_tensors="pt")

    # Generate response
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Update history
    st.session_state["history"].append(user_input)
    st.session_state["history"].append(response)

    # Display bot reply
    st.write(f"**Bot:** {response}")

# Show conversation history
if st.session_state["history"]:
    st.markdown("---")
    st.write("### Conversation History")
    for i, msg in enumerate(st.session_state["history"]):
        speaker = "You" if i % 2 == 0 else "Bot"
        st.write(f"**{speaker}:** {msg}")
