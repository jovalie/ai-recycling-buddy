import streamlit as st
from dotenv import load_dotenv
import os
import base64
import sys
import requests

# Add the src directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage, AIMessage

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(current_script_dir)
media_dir = os.path.join(repo_dir, "img")

st.set_page_config(
    page_title="SusTech Recycling Agent: Your Guide to Sustainable Recycling",
    page_icon="♻️",
    layout="wide",
)

examples = [
    "What are the recycling laws for plastic bottles in California?",
    "How do I properly recycle electronics in Germany?",
    "What materials are considered hazardous waste in the US?",
    "Can I recycle pizza boxes? What about the grease stains?",
    "What's the difference between recycling programs in the US vs Germany?",
]

# Sidebar: SusTech Recycling Mission
with st.sidebar:
    st.title("♻️ AI Recycling Buddy")

    # Region selector (Germany default) as dropdown at the top
    region = st.selectbox("Region", options=["Germany", "US"], index=0, key="region_select")

    st.subheader("Our Mission 🌱")

    # Region selector (Germany default) placed at the top of the sidebar for quick access
    st.write(
        """
    We are dedicated to promoting sustainable recycling practices and educating communities about proper waste management.

    Our AI assistant provides:
    - 📚 Accurate information on recycling laws and regulations
    - 🇺🇸 US and 🇩🇪 German recycling guidelines
    - 🔍 Answers to common recycling questions
    - 💡 Tips for sustainable living
    - ♻️ Guidance on proper sorting and disposal

    Together, let's build a more sustainable future!
    """
    )

    st.subheader("Focus Areas")
    st.write(
        """
    - 📋 Recycling Laws & Regulations
    - 🗂️ Proper Waste Sorting
    - ⚠️ Hazardous Materials
    - 🌍 Cross-border Recycling
    - 💡 Sustainable Living Tips
    """
    )

    # Reset conversation button
    if st.button("Reset Conversation 🔄", use_container_width=True, key="reset_button"):
        st.session_state.chat_history = []
        st.rerun()


def get_image_base64(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        # Return a placeholder if image doesn't exist
        return ""


# Try to load images, use placeholders if they don't exist
try:
    logo_base64 = get_image_base64(f"{media_dir}/minilogo.png")
    img_base64 = get_image_base64(f"{media_dir}/logo.png")
    img_bg = get_image_base64(f"{media_dir}/bg.jpg")
except:
    logo_base64 = ""
    img_base64 = ""
    img_bg = ""

# Main logo display
if img_base64:
    with st.container():
        st.markdown(
            f"""
            <div style='display: flex; justify-content: center; align-items: center; margin-top: -3rem; margin-bottom: 0rem;'>
                <img src='data:image/png;base64,{img_base64}' width='300'/>
            </div>
        """,
            unsafe_allow_html=True,
        )

# Custom CSS for recycling theme
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]{{
    background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 50%, #CDDC39 100%);
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"][data-testid="stChatInputTextArea"]{{
    background: rgba(255, 255, 255, 0.1) !important;
}}

input[type="text"] {{
    background-color: rgba(255, 255, 255, 0.9) !important;
    color: #2E7D32 !important;
    border-radius: 10px !important;
    border: 2px solid #4CAF50 !important;
    font-weight: 500 !important;
}}

button[kind="primary"] {{
    background-color: #4CAF50 !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}}

button[kind="primary"]:hover {{
    background-color: #388E3C !important;
    transform: translateY(-2px) !important;
}}

[data-testid="stBottomBlockContainer"]{{
    background-color: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(8px);
    border-radius: 15px;
    padding: 15px;
    margin-top: 10px;
    border: 2px solid rgba(76, 175, 80, 0.3);
}}

[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-right: 3px solid #4CAF50 !important;
}}

.stChatMessage {{
    background-color: rgba(255, 255, 255, 0.9) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    border: 1px solid rgba(76, 175, 80, 0.2) !important;
}}

.stChatMessage[data-testid="stChatMessage-human"] {{
    background-color: rgba(76, 175, 80, 0.1) !important;
    border: 2px solid #4CAF50 !important;
}}

.stChatMessage[data-testid="stChatMessage-assistant"] {{
    background-color: rgba(255, 255, 255, 0.95) !important;
    border: 2px solid #8BC34A !important;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# API configuration
API_URL = "http://localhost:8000/query"


def generate_answer(prompt: str, chat_history: list = None) -> tuple[str, dict]:
    """Generate answer using the FastAPI backend."""
    if chat_history is None:
        chat_history = []

    # include selected region from sidebar (default to Germany)
    selected_region = st.session_state.get("region_select", "Germany")
    payload = {"question": prompt, "chat_history": chat_history, "region": selected_region}

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["response"], data["usage"]
    except requests.RequestException as e:
        print(f"❌ Error talking to backend: {e}")
        return "Sorry, something went wrong. Please make sure the FastAPI server is running.", {}


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Intro message
if not st.session_state.chat_history:
    with st.chat_message("ai"):
        st.markdown(
            """
        **♻️ Hello! I'm your AI Recycling Buddy – your guide to sustainable recycling practices in the US and Germany.**

        I can help you with:
        - 🇺🇸 US recycling laws and regulations by state
        - 🇩🇪 German recycling guidelines and proper waste sorting
        - ♻️ How to recycle different materials correctly
        - ⚠️ Information about hazardous waste disposal
        - 💡 Tips for reducing waste and living sustainably

        What recycling question can I help you with today?
        """
        )

# Chat input
user_query = st.chat_input("Ask me about recycling in the US or Germany...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("ai"):
        with st.spinner("🔍 Researching recycling information..."):
            try:
                # Use the API to get response and usage data
                response, usage = generate_answer(str(user_query), [])

                st.markdown(response)

                # Display usage information in expandable JSON format
                # with st.expander("📊 Full Agent Pipeline Data", expanded=False):
                #     st.json(usage)

                print(f"Final answer: {response}")
                print(f"Usage data keys: {list(usage.keys()) if isinstance(usage, dict) else 'Not a dict'}")
                st.session_state.chat_history.append(AIMessage(response))

            except Exception as e:
                error_msg = f"I apologize, but I encountered an error while processing your recycling question. Please try again or rephrase your question. Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(error_msg))
