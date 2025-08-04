import os
import base64
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from config import PDFConfig, RagConfig
from rag_pipeline import InMemoryRagPipeline

load_dotenv()

st.title("📄 RAG PDF Q&A App")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.pdf_name = None

if uploaded_file is not None:
    pdf_path = os.path.join(os.getcwd(), uploaded_file.name)

    if st.session_state.pdf_name != uploaded_file.name:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        pdf_config = PDFConfig(filename=uploaded_file.name)
        rag_config = RagConfig(pdf_config=pdf_config)
        st.session_state.rag = InMemoryRagPipeline(rag_config)
        st.session_state.rag.setup()
        st.session_state.pdf_name = uploaded_file.name

        try:
            os.remove(pdf_path)
        except PermissionError:
            st.warning("Could not delete the uploaded file immediately. It may be in use.")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        response = st.session_state.rag.run(question)

        st.subheader("Response")
        st.markdown(response.get("response", "No response"))

        st.subheader("Context Texts")
        for text in response.get("context", {}).get("texts", []):
            st.write(text)
            st.divider()

        st.subheader("Context Images")
        for image_b64 in response.get("context", {}).get("images", []):
            try:
                image_bytes = base64.b64decode(image_b64)
                st.image(BytesIO(image_bytes), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not decode image: {e}")
