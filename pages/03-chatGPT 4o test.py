import streamlit as st
import openai
import pandas as pd
from PIL import Image
import pytesseract  # OCR을 위한 라이브러리
import settings
import os
import pickle

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

openai.api_key = st.session_state.api_key

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

# 파일 업로드 기능을 채팅 탭으로 이동
uploaded_file = tab1.file_uploader("Upload your file", type=["csv", "xlsx", "png", "jpg", "jpeg"])

if uploaded_file:
    data_str = ""
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
        data_str = pytesseract.image_to_string(img)
        if not data_str.strip():
            data_str = "OCR could not extract text from the image."
    else:
        data_str = "Unsupported file type."

    if data_str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Analyze the following data:\n\n{data_str}"}
                ]
            )
            st.write("Analysis Result:")
            st.write(response.choices[0].message['content'].strip())
        except Exception as e:
            st.error(f"Error processing file: {e}")

if prompt := st.chat_input():
    add_history("user", prompt)
    tab1.chat_message("user").write(prompt)
    with tab1.chat_message("assistant"):
        msg = st.empty()
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        st.session_state.history.append((prompt, response.choices[0].message['content']))
        add_history("ai", response.choices[0].message['content'])
        msg.markdown(response.choices[0].message['content'])

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
