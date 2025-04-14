import os
import requests
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


load_dotenv()
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")


qdrant = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM Retrieval Augmented System

def rag_app():
    st.title("LLM Retrieval Augmented System")
    st.write("Ask questions based on the Academic City Student Handbook document.")

    file_path = "handbook.pdf"

    @st.cache_data(show_spinner=False)
    def extract_text_from_pdf(file_path):
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    @st.cache_data(show_spinner=False)
    def split_text(text, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    @st.cache_data(show_spinner=False)
    def embed_chunks(chunks):
        return np.array(embedding_model.encode(chunks))

    @st.cache_data(show_spinner=False)
    def process_pdf_and_index():
        pdf_text = extract_text_from_pdf(file_path)
        chunks = split_text(pdf_text)
        embeddings = embed_chunks(chunks)

        # Recreate and upload
        qdrant.recreate_collection(
            collection_name="handbook",
            vectors_config=VectorParams(
                size=embeddings.shape[1],
                distance=Distance.COSINE
            ),
        )
        qdrant.upload_points(
            collection_name="handbook",
            points=[PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": chunks[i]})
                    for i in range(len(chunks))]
        )
        return True
    
    # pdf read confirmation
    
    if process_pdf_and_index():
        st.success("PDF processed and indexed!")

    
    query_text = st.text_input("Ask a question about the document:")
     #query embedding
    if query_text:
        def retrieve_relevant_chunks(query, top_k=5):
            query_embedding = embedding_model.encode([query])[0]
            results = qdrant.search(
                collection_name="handbook",
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            return [hit.payload["text"] for hit in results]

        relevant_chunks = retrieve_relevant_chunks(query_text)

        def generate_answer(query, retrieved_text):
            prompt = (
                "You are an AI assistant that answers questions based on provided document excerpts. "
                "Your goal is to extract the most relevant information and provide a concise, factual summary.\n\n"
                "### Document Excerpts:\n"
                f"{retrieved_text}\n\n"
                "### User Question:\n"
                f"{query}\n\n"
                "### Instructions:\n"
                "- Identify the key points that directly answer the user's question.\n"
                "- Provide a *clear and structured summary* in *2-3 sentences*.\n"
                "- Avoid unnecessary details or repeating the text verbatim.\n"
                "- If the excerpts do not contain enough information, state: 'The document does not provide a clear answer to this question.'\n\n"
                "### Answer:"
            )

            try:
                response = requests.post(
                    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                    headers={"Authorization": f"Bearer {HF_API_KEY}"},
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "temperature": 0.3,
                            "max_new_tokens": 200
                        }
                    }
                )
                result = response.json()
                if isinstance(result, list):
                    generated_text = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    generated_text = result.get("generated_text", "")
                else:
                    return " Error: Unexpected response format from model."

                return generated_text.split("### Answer:")[-1].strip()
            except Exception as e:
                return f" Error generating answer: {e}"

        # Show final answer
        answer = generate_answer(query_text, "\n".join(relevant_chunks))
        st.markdown("###  Answer:")
        st.write(answer)

    else:
        st.info("👆 Ask a question about the document.")


# Regression Explorer
def regression_app():
    st.title("Regression Explorer")
    st.markdown("Upload a dataset with a **continuous target column** to perform linear regression.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="reg_csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        if st.checkbox("Drop rows with missing values"):
            df = df.dropna()

        if st.checkbox("Fill missing values with mean"):
            df = df.fillna(df.mean(numeric_only=True))

        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            encoding_method = st.selectbox("Choose Encoding Method for Categorical Variables", ["One-Hot Encoding", "Label Encoding"])
            if encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, drop_first=True)
            elif encoding_method == "Label Encoding":
                label_encoder = LabelEncoder()
                for col in categorical_columns:
                    df[col] = label_encoder.fit_transform(df[col])

        st.subheader("Data after Preprocessing")
        st.dataframe(df.head())

        all_columns = df.columns.tolist()
        target_column = st.selectbox("Select Target Column (to predict)", all_columns)
        feature_columns = st.multiselect(" Select Feature Columns", [col for col in all_columns if col != target_column])

        if st.button("Train Regression Model"):
            if not feature_columns:
                st.warning("Please select at least one feature column.")
            else:
                X = df[feature_columns]
                y = df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.success("Model Trained Successfully")
                st.markdown(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f}")
                st.markdown(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                ax.plot(y_test, y_test, color='red')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

                st.subheader("🔍 Make Custom Prediction")
                custom_input = {col: st.number_input(f"Enter {col}", value=float(X[col].mean())) for col in feature_columns}
                if st.button("Predict Custom Input"):
                    prediction = model.predict(pd.DataFrame([custom_input]))[0]
                    st.success(f"📊 Predicted {target_column}: {prediction:.2f}")

# Clustering Explorer
def clustering_app():
    st.title("📊 Clustering Explorer")
    st.markdown("Upload a dataset with multiple features to perform K-Means clustering.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="cluster_csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop='first'), categorical_cols)
        ])

        df_processed = preprocessor.fit_transform(df)
        df_processed = pd.DataFrame(df_processed)

        st.subheader(" Data after Preprocessing")
        st.dataframe(df_processed.head())

        num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_processed)

        st.subheader(f" Clustering Results (k={num_clusters})")
        st.dataframe(df[['Cluster'] + df.columns.tolist()].head())

        st.subheader("🗺 Cluster Visualization")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_processed)
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Cluster'], cmap='viridis')
        fig.colorbar(scatter)
        st.pyplot(fig)

#sidebar
app = st.sidebar.selectbox("Select model to use", ["LLM Retrieval Augmented System", "Regression Explorer", "Clustering Explorer", "Neural Network Classifier"])

if app == "LLM Retrieval Augmented System":
    rag_app()
elif app == "Regression Explorer":
    regression_app()
elif app == "Clustering Explorer":
    clustering_app()
else:
    neural_app()
