# Rough Chatbot Project

This project is a simple and modular chatbot application built using Python. It features different UI frontends (Gradio and Streamlit) and uses pre-trained language models from Hugging Face for generating responses.

---

## 🔧 Features

- Basic terminal-based chatbot (`chatbot.py`)
- Gradio GUI chatbot interface (`gradio_chatbot.py`, `gradio_chatbot2.py`)
- Advanced Gradio interface using FLAN-T5 model (`gradio_FLAN_T5.py`)
- Streamlit chatbot interface (`streamlit_chatbot.py`)

---

## 🧠 Models Used

| Script                | Model Used                               |
|-----------------------|-------------------------------------------|
| `chatbot.py`          | None (rule-based template response)       |
| `gradio_chatbot.py`   | `Helsinki-NLP/opus-mt-en-ROMANCE`         |
| `gradio_chatbot2.py`  | `t5-base` (from Hugging Face Transformers)|
| `gradio_FLAN_T5.py`   | `google/flan-t5-small`                    |
| `streamlit_chatbot.py`| `google/flan-t5-small`                    |

---

## 📦 Requirements

Install all necessary Python packages with:

```bash
pip install -r requirements.txt
