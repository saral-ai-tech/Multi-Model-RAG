
import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fitz
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from openai import RateLimitError

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

OPENAI_API_KEY = ""


FAISS_DIR = "faiss_index"
UPLOAD_DIR = "uploads"

os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------
# SESSION
# ------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------
# RATE LIMIT SAFE CALL
# ------------------------------------------------


def safe_call(llm, msg):

    retry = 0

    while retry < 5:

        try:
            return llm.invoke(msg)

        except RateLimitError:

            time.sleep(2**retry)
            retry += 1

    raise Exception("Rate limit failed")


# ------------------------------------------------
# PDF EXTRACT
# ------------------------------------------------


def extract_pdf(pdf_path):

    text = ""
    images = []

    doc = fitz.open(pdf_path)

    for p in range(len(doc)):

        page = doc[p]

        text += page.get_text()

        for i, img in enumerate(page.get_images(full=True)):

            xref = img[0]
            data = doc.extract_image(xref) # getting images

            path = f"{UPLOAD_DIR}/p{p}_{i}.png"

            open(path, "wb").write(data["image"])

            images.append(path)

    return text, images  


# ------------------------------------------------
# TEXT SPLIT
# ------------------------------------------------


def split_text(text):

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    return splitter.split_text(text)


# ------------------------------------------------
# PARALLEL IMAGE CAPTION
# ------------------------------------------------


def caption_image(img_path):

    cache = img_path + ".txt"

    if os.path.exists(cache):

        return open(cache).read()

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

    with open(img_path, "rb") as f:

        img64 = base64.b64encode(f.read()).decode()

    msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe image for semantic search"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img64}"},
                },
            ],
        }
    ]

    res = safe_call(llm, msg)

    open(cache, "w").write(res.content)

    return res.content


def parallel_caption(images, progress):

    results = []

    with ThreadPoolExecutor(max_workers=3) as ex:

        futures = [ex.submit(caption_image, img) for img in images[:10]]

        done = 0

        for f in futures:

            results.append(f.result())

            done += 1

            progress.progress(done / len(futures))

    return results


# ------------------------------------------------
# CREATE VECTOR STORE
# ------------------------------------------------


def create_index(chunks, images):

    progress = st.progress(0.0)

    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    docs = [Document(page_content=c) for c in chunks]

    progress.progress(0.3)

    captions = parallel_caption(images, progress)

    for cap in captions:

        docs.append(Document(page_content=cap))

    progress.progress(0.8)

    vs = FAISS.from_documents(docs, embedding)

    vs.save_local(FAISS_DIR)

    progress.progress(1.0)


# ------------------------------------------------
# LOAD VECTOR STORE
# ------------------------------------------------


def load_index():

    if not os.path.exists(f"{FAISS_DIR}/index.faiss"):
        return None

    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    return FAISS.load_local(FAISS_DIR, embedding, allow_dangerous_deserialization=True)


# ------------------------------------------------
# CHAT
# ------------------------------------------------


def ask(question):

    vs = load_index()

    docs = vs.similarity_search(question, k=4)

    context = "\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

    history = ""

    for q, a in st.session_state.chat_history:
        history += f"User:{q}\nAssistant:{a}\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are multimodal PDF assistant"),
            ("human", "History:{history}\nContext:{context}\nQuestion:{question}"),
        ]
    )

    chain = prompt | llm

    res = safe_call(
        chain, {"history": history, "context": context, "question": question}
    )

    answer = res.content

    st.session_state.chat_history.append((question, answer))


# ------------------------------------------------
# UI
# ------------------------------------------------

st.title("🔥 FAST Multimodal RAG")

tab1, tab2 = st.tabs(["Index PDF", "Chat"])

# INDEX TAB
with tab1:

    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file:

        path = os.path.join(UPLOAD_DIR, file.name)

        open(path, "wb").write(file.getbuffer())

        text, images = extract_pdf(path) # get pdf text and images

        chunks = split_text(text) # text

        create_index(chunks, images)

        st.success("Index complete")

# CHAT TAB
with tab2:

    if not os.path.exists(f"{FAISS_DIR}/index.faiss"):

        st.warning("Please index PDF first")

    else:

        q = st.text_input("Ask")

        if st.button("Send"):

            ask(q)

        for q, a in st.session_state.chat_history:

            st.write("You:", q)
            st.write("AI:", a)
