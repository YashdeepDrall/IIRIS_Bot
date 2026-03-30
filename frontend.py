# frontend.py
import streamlit as st
import requests
import uuid
import os
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="IIRIS Cybersecurity Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- API Endpoint ---
# This should point to the address where your chatbot.py backend is running

API_URL = os.getenv("API_URL", "http://127.0.0.1:10000/ask")

# --- Sidebar for Model Settings ---
st.sidebar.title("⚙️ Model Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, help="Controls the randomness of the output. Lower is more deterministic.")
max_tokens = st.sidebar.slider("Max Tokens", min_value=256, max_value=8192, value=512, step=256, help="The maximum number of tokens to generate in the response.")
k = st.sidebar.slider("Retrieved Documents (k)", min_value=1, max_value=100, value=15, step=1, help="The number of relevant documents to retrieve for context.")

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Cybersecurity Consultant at IIRIS. How can I assist you?"}
    ]

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    print(f"New user session: {st.session_state.user_id}")

# --- UI Components ---
st.title("IIRIS Cybersecurity Consultant 🤖")
st.write("Ask me anything about IIRIS, our services, or general cybersecurity topics.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and API Interaction ---
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response while waiting
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call the backend API
            # Prepare history (excluding the current prompt which is already in 'question')
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            
            payload = {
                "question": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "k": k,
                "user_id": st.session_state.user_id,
                "history": history
            }
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (like 404 or 500)
            
            data = response.json()
            full_response = data["answer"]
            message_placeholder.markdown(full_response)
            
            if "usage" in data:
                usage = data["usage"]
                st.caption(f"📊 Tokens Used: {usage.get('total_tokens', 0)} (Prompt: {usage.get('prompt_tokens', 0)}, Completion: {usage.get('completion_tokens', 0)})")
            
            # Add the final assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            error_message = f"Sorry, I couldn't connect to the server. Please ensure the backend is running.\n\n*Error: {e}*"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
