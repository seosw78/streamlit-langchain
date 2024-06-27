import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler
import os
import pickle
import pandas as pd
from PIL import Image
import settings

# API KEY 설정
if "api_key" not in st.session_state:
    config = settings.load_config()
    if "api_key" in config:
        st.session_state.api_key = config["api_key"]
    else:
        st.session_state.api_key = ""

st.title("서울 선생님들 테스트용 ChatGPT")

st.markdown(
    f"""API KEY
    `{st.session_state.api_key[:-15] + '***************'}`
    """
)

if "history" not in st.session_state:
    st.session_state.history = []

if "user" not in st.session_state:
    st.session_state.user = []

if "ai" not in st.session_state:
    st.session_state.ai = []

def add_history(role, content):
    if role == "user":
        st.session_state.user.append(content)
    elif role == "ai":
        st.session_state.ai.append(content)

model_name = st.empty()
tab1, tab2 = st.tabs(["Chat", "Settings"])

class StreamCallback(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.full_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.full_text += token
        self.container.markdown(self.full_text)

# ChatOpenAI 객체 생성
try:
    llm = ChatOpenAI(
        model="gpt-4",
        streaming=True,
        callbacks=[StreamCallback(st.empty())],
        api_key=st.session_state.api_key,
    )
except Exception as e:
    st.error(f"Error initializing ChatOpenAI: {e}")

# ConversationChain 객체 생성
conversation = ConversationChain(
    llm=llm, verbose=False, memory=ConversationBufferMemory()
)

prompt_preset = "서울 선생님들 환영합니다."
prompt_input = prompt_preset

def create_prompt_template(prompt_input):
    prompt_template = PromptTemplate.from_template(
        """
{custom_prompt}

HISTORY:
{history}

QUESTION:
{input}

ANSWER:

"""
    )
    prompt_template = prompt_template.partial(custom_prompt=prompt_input)
    return prompt_template

if prompt_input:
    prompt_template = create_prompt_template(prompt_input)
    conversation.prompt = prompt_template

model_input = tab2.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], index=1)

if model_input:
    settings.save_config({"model": model_input})
    llm.model_name = model_input
    model_name.markdown(f"#### {model_input}")

def print_history():
    for i in range(len(st.session_state.ai)):
        tab1.chat_message("user").write(st.session_state["user"][i])
        tab1.chat_message("ai").write(st.session_state["ai"][i])

def save_chat_history(title):
    pickle.dump(
        st.session_state.history,
        open(os.path.join("./chat_history", f"{title}.pkl"), "wb"),
    )

def load_chat_history(filename):
    with open(os.path.join("./chat_history", f"{filename}.pkl"), "rb") as f:
        st.session_state.history = pickle.load(f)
        st.session_state.user.clear()
        st.session_state.ai.clear()
        for user, ai in st.session_state.history:
            add_history("user", user)
            add_history("ai", ai)

def load_chat_history_list():
    files = os.listdir("./chat_history")
    files = [f.split(".")[0] for f in files]
    return files

with st.sidebar:
    clear_btn = st.button("대화내용 초기화", type="primary", use_container_width=True)
    save_title = st.text_input("저장할 제목")
    save_btn = st.button("대화내용 저장", use_container_width=True)

    if clear_btn:
        st.session_state.history.clear()
        st.session_state.user.clear()
        st.session_state.ai.clear()
        print_history()

    if save_btn and save_title:
        save_chat_history(save_title)

    selected_chat = st.selectbox("대화내용 불러오기", load_chat_history_list(), index=None)
    load_btn = st.button("대화내용 불러오기", use_container_width=True)
    if load_btn and selected_chat:
        load_chat_history(selected_chat)

print_history()

uploaded_file = tab2.file_uploader("Upload your file", type=["csv", "xlsx", "png", "jpg", "jpeg"])

if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded Excel file:")
        st.write(df)
        data_str = df.to_string(index=False)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file:")
        st.write(df)
        data_str = df.to_string(index=False)
    elif uploaded_file.name.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        data_str = "An image file has been uploaded. GPT-4 cannot process image content."
    else:
        data_str = "Unsupported file type."

    if data_str:
        try:
            response = llm.completion(
                prompt=f"Analyze the following data:\n\n{data_str}"
            )
            st.write("Analysis Result:")
            st.write(response.choices[0].text.strip())
        except Exception as e:
            st.error(f"Error processing file: {e}")

if prompt := st.chat_input():
    add_history("user", prompt)
    tab1.chat_message("user").write(prompt)
    with tab1.chat_message("assistant"):
        msg = st.empty()
        llm.callbacks[0].container = msg
        for user, ai in st.session_state.history:
            conversation.memory.save_context(inputs={"human": user}, outputs={"ai": ai})
        response = conversation.invoke(
            {"input": prompt, "history": st.session_state.history}
        )
        st.session_state.history.append((prompt, response["response"]))
        add_history("ai", response["response"])
