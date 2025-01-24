import os
from dotenv import load_dotenv
import openai
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OpenAI API key not found. Set it in the .env file or as an environment variable.")

# Initialize the model
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    corpus = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                corpus.append((filename, text))
    return corpus

# Clean retrieved text chunks
def clean_text(text):
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())  # Normalize spaces
    return text

# Preprocess PDFs and create embeddings
@st.cache_resource
def preprocess_and_index(pdf_folder):
    corpus = extract_text_from_pdfs(pdf_folder)
    texts = [clean_text(doc[1]) for doc in corpus]
    embeddings = retrieval_model.encode(texts, convert_to_tensor=False)

    # Create FAISS index
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, texts

# Load data
pdf_folder = "./"  # Folder containing the PDFs
index, corpus = preprocess_and_index(pdf_folder)

# Retrieve relevant chunks
def retrieve_chunks(query):
    query_embedding = retrieval_model.encode([query], convert_to_tensor=False)
    _, indices = index.search(query_embedding, k=3)
    results = [corpus[i] for i in indices[0]]
    return results

# Generate a response using GPT-3.5 (restricted to financial topics)
def generate_gpt3_response(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)  # Combine chunks for context
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only answers questions related to financial topics. If the question is unrelated to financial topics, politely refuse to answer."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    answer = response.choices[0].message.content

    # Check if the answer is relevant to financial topics
    if "I cannot answer" in answer or "unrelated" in answer.lower():
        return "I'm sorry, I can only answer questions related to financial topics."
    return answer

# Streamlit App
st.title("EduMaster's Financial Literacy Chatbot")

# Input options
input_type = st.radio("Choose Input Method:", ("Text", "Voice"))
query = ""

# Text input form
if input_type == "Text":
    with st.form("query_form"):
        query = st.text_input("Ask your question:")
        submitted = st.form_submit_button("Submit")
        if submitted and query:
            with st.spinner("Processing..."):
                chunks = retrieve_chunks(query)
                response = generate_gpt3_response(query, chunks)
                st.write("### Chatbot Response:")
                st.write(response)

                # Text-to-Speech for response
                tts = gTTS(response, lang="en")
                tts.save("response.mp3")
                st.audio("response.mp3", format="audio/mp3")

# Voice input
elif input_type == "Voice":
    if st.button("Record Voice"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Recording...")
            try:
                audio = recognizer.listen(source, timeout=5)
                query = recognizer.recognize_google(audio, language="en-IN")
                st.success(f"You said: {query}")
                
                if query:
                    with st.spinner("Processing..."):
                        chunks = retrieve_chunks(query)
                        response = generate_gpt3_response(query, chunks)
                        st.write("### Chatbot Response:")
                        st.write(response)

                        # Text-to-Speech for response
                        tts = gTTS(response, lang="en")
                        tts.save("response.mp3")
                        st.audio("response.mp3", format="audio/mp3")
            except Exception as e:
                st.error(f"Error: {e}")
