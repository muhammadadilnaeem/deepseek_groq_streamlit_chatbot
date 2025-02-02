
# Import libraries
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)

# Load environment variables
load_dotenv()

# Set up Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Set page configuration (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config("DeepSeek Code Companion", page_icon="🧠", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* Additional styling for chat messages */
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-message.user {
        background-color: #4a4a4a;
    }
    .chat-message.bot {
        background-color: #3d3d3d;
    }
</style>
""", unsafe_allow_html=True)

# App title and caption
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1-distill-llama-70b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Groq](https://groq.com/) | [LangChain](https://python.langchain.com/)")

# Initiate chat engine
groq_deepseek_engine = ChatGroq(
    api_key=groq_api_key,
    model=selected_model,
    temperature=0.3,
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Streamlit session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{
        "role": "ai",
        "content": "Hi! I'm Adil's Demo DeepSeek Bot. How can I help you code today? 💻",
        "think": None  # New field to store thinking content
    }]

# Chat container for Streamlit
chat_container = st.container()

# ========== KEY CHANGE: INPUT BOX ABOVE MESSAGES ==========
# Chat input at the top (above messages)
user_query = st.chat_input("Type Your Coding Question Here 😎")

# Display chat messages below the input
with chat_container:
    for message in st.session_state.message_log:
        # ========== KEY CHANGE: THINKING EXPANDER ABOVE MESSAGE ==========
        if message["think"]:
            with st.expander("🧠 DeepSeek Thinking"):
                st.markdown(message["think"])
        
        # Display the message content
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Function to get DeepSeek response
def generated_ai_response(prompt_chain):
    preprocessing_pipeline = prompt_chain | groq_deepseek_engine | StrOutputParser()
    return preprocessing_pipeline.stream({})

# Define a function to build a prompt chain
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"])) 
    return ChatPromptTemplate.from_messages(prompt_sequence)

# If user sends a query
if user_query:
    # Add user message to the log
    st.session_state.message_log.append({"role": "user", "content": user_query, "think": None})

    # Generate AI response
    with st.spinner("🧠 Thinking..."):
        prompt_chain = build_prompt_chain()
        full_response = ""
        think_content = ""
        chat_content = ""
        
        # Stream response
        response_placeholder = st.empty()
        for chunk in generated_ai_response(prompt_chain):
            full_response += chunk
            response_placeholder.markdown(full_response)

        # Parse response
        if "<think>" in full_response and "</think>" in full_response:
            think_content = full_response.split("<think>")[1].split("</think>")[0].strip()
            chat_content = full_response.split("</think>")[1].strip()
        else:
            chat_content = full_response

        # Add AI response to log WITH THINKING SEPARATED
        st.session_state.message_log.append({
            "role": "ai",
            "content": chat_content,
            "think": think_content if think_content else None
        })

    # Rerun to update chat display
    st.rerun()
