import streamlit as st
import google.generativeai as genai
import os
import time
from typing import Generator

# Configure Gemini API
def configure_gemini():
    """Configure the Gemini API with the API key from environment variables"""
    api_key = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "gpt_name" not in st.session_state:
        st.session_state.gpt_name = "Custom GPT"
    
    if "gpt_description" not in st.session_state:
        st.session_state.gpt_description = "A helpful AI assistant"
    
    if "gpt_instructions" not in st.session_state:
        st.session_state.gpt_instructions = "You are a helpful AI assistant. Respond to user queries in a helpful and informative manner."
    
    if "model" not in st.session_state:
        try:
            st.session_state.model = configure_gemini()
        except Exception as e:
            st.error(f"Failed to configure Gemini API: {str(e)}")
            st.session_state.model = None

def stream_response(response) -> Generator[str, None, None]:
    """Stream the response from Gemini API"""
    try:
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error generating response: {str(e)}"

def get_ai_response(messages: list, instructions: str) -> str:
    """Get response from Gemini API"""
    if st.session_state.model is None:
        return "Error: Gemini API not configured properly. Please check your API key."
    
    try:
        # Prepare the conversation context
        conversation_context = f"Instructions: {instructions}\n\n"
        
        # Add conversation history
        for msg in messages[-10:]:  # Keep last 10 messages for context
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
        
        # Get the latest user message
        latest_message = messages[-1]["content"]
        conversation_context += f"Human: {latest_message}\nAssistant:"
        
        # Generate response
        response = st.session_state.model.generate_content(
            conversation_context,
            stream=True
        )
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def render_sidebar():
    """Render the sidebar with GPT configuration options"""
    with st.sidebar:
        st.header("🤖 Custom GPT Configuration")
        
        # GPT Name
        st.session_state.gpt_name = st.text_input(
            "GPT Name",
            value=st.session_state.gpt_name,
            help="Give your custom GPT a name"
        )
        
        # GPT Description
        st.session_state.gpt_description = st.text_area(
            "Description",
            value=st.session_state.gpt_description,
            height=100,
            help="Describe what your GPT does"
        )
        
        # GPT Instructions
        st.session_state.gpt_instructions = st.text_area(
            "Instructions",
            value=st.session_state.gpt_instructions,
            height=200,
            help="Provide detailed instructions for your GPT's behavior"
        )
        
        st.divider()
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # API Status
        st.subheader("📊 Status")
        if st.session_state.model:
            st.success("✅ Gemini API Connected")
        else:
            st.error("❌ Gemini API Not Connected")
            st.caption("Check your GEMINI_API_KEY environment variable")

def render_chat_message(message: dict):
    """Render a single chat message"""
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])

def render_chat_interface():
    """Render the main chat interface"""
    # Display GPT info
    st.title(f"💬 {st.session_state.gpt_name}")
    st.caption(st.session_state.gpt_description)
    
    # Display conversation history
    for message in st.session_state.messages:
        render_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to conversation
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        render_chat_message(user_message)
        
        # Generate AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                # Get streaming response
                response_stream = get_ai_response(
                    st.session_state.messages, 
                    st.session_state.gpt_instructions
                )
                
                if isinstance(response_stream, str):
                    # Error case
                    st.write(response_stream)
                    ai_message = {"role": "assistant", "content": response_stream}
                else:
                    # Streaming response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        for chunk in stream_response(response_stream):
                            full_response += chunk
                            response_placeholder.write(full_response + "▊")
                            time.sleep(0.01)  # Small delay for smooth streaming effect
                        
                        response_placeholder.write(full_response)
                        ai_message = {"role": "assistant", "content": full_response}
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        response_placeholder.write(error_msg)
                        ai_message = {"role": "assistant", "content": error_msg}
        
        # Add AI response to conversation
        st.session_state.messages.append(ai_message)
        st.rerun()

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Custom GPT",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.caption("Powered by Google Gemini API • Built with Streamlit")

if __name__ == "__main__":
    main()
