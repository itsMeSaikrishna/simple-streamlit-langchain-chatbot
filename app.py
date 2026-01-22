import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Simple Chatbot", page_icon="💬")
st.title("Simple Chatbot")


@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

SYSTEM_PROMPT = "You are a helpful AI assistant"


if "chat" not in st.session_state:
    st.session_state.chat = [SystemMessage(content=SYSTEM_PROMPT)]

if st.button("Clear Chat"):
    st.session_state.chat = [SystemMessage(content=SYSTEM_PROMPT)]
    st.rerun()


for msg in st.session_state.chat:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)


user_input = st.chat_input("You:")

if user_input:
    st.session_state.chat.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = model.invoke(st.session_state.chat)
            st.write(result.content)

    st.session_state.chat.append(AIMessage(content=result.content))
