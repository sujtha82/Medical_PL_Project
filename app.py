# app.py
import gradio as gr
from main import get_answer
chatbot = gr.Chatbot(height=400, type="messages")
def chat_interface(query, history):
    # ensure history is a list we can append to
    if history is None:
        history = []
    if not query.strip():
        return history, ""
    response = get_answer(query)
    history.append((query, response))
    # return updated chat history and clear the input box
    return history, ""

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("<h1 style='text-align:center;'>🏥 Medical Claims Hybrid RAG + ReRanker Chatbot</h1>")
    
    chatbot = gr.Chatbot(height=400)
    user_input = gr.Textbox(label="Ask about medical claims", placeholder="e.g. Why was my claim denied?", lines=2)
    send_button = gr.Button("Send")
    clear = gr.Button("Clear Chat")

    user_input.submit(chat_interface, [user_input, chatbot], [chatbot, user_input])
    send_button.click(chat_interface, [user_input, chatbot], [chatbot, user_input])
    clear.click(lambda: [], None, chatbot, queue=False)
    
demo.launch()