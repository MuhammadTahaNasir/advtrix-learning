import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# =====================
# Terrag AI - Market Assistance
# =====================
st.set_page_config(page_title="Terrag AI - Market Assistance", layout="wide")

st.title("📊 Terrag AI - Market Assistance")
st.subheader("Upload CSV, Excel, or PDF — then explore lead scoring, segmentation, forecasting and chat with your data!")

st.sidebar.header("📁 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV, Excel, or PDF file", type=["csv", "xlsx", "xls", "pdf"])

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------- PDF Reader Helper ----------
def read_pdf(file):
    try:
        import fitz  # PyMuPDF
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
        return pd.DataFrame({"Text": text.split("\n")})
    except:
        st.error("❌ Unable to read PDF. Please upload a table-based file.")
        return None

# ---------- Smart Q&A Function ----------
def analyze_question(df, question):
    lower_q = question.lower()
    best_answer = "Sorry, I couldn't confidently find the exact answer, but here’s something relevant."

    for col in df.columns:
        if col.lower() in lower_q:
            if any(k in lower_q for k in ["most", "highest", "top", "max"]):
                try:
                    top_value = df[col].max()
                    top_row = df[df[col] == top_value].head(1)
                    return (f"🔍 The highest `{col}` is **{top_value}**.", top_row)
                except:
                    continue
            elif any(k in lower_q for k in ["average", "mean", "avg"]):
                try:
                    avg_val = df[col].mean()
                    return (f"📊 The average `{col}` is **{avg_val:.2f}**.", None)
                except:
                    continue

    # Fallback semantic match
    row_texts = df.astype(str).agg(" ".join, axis=1).tolist()
    row_embeddings = model.encode(row_texts, show_progress_bar=False)
    question_embedding = model.encode([question])
    sims = cosine_similarity(question_embedding, row_embeddings)[0]
    top_index = sims.argmax()
    top_row = df.iloc[[top_index]]

    return (f"💡 Based on your question, here’s the most relevant data row:", top_row)

# ---------- PDF Export Helper ----------
def convert_df_to_pdf(df):
    fig, ax = plt.subplots(figsize=(12, min(20, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf

# ---------- Chart in Chat ----------
def render_chart_in_chat(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts().head(10).plot(kind='bar', ax=ax)
    st.pyplot(fig)

# ---------- App Logic ----------
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type in ["csv"]:
        df = pd.read_csv(uploaded_file)
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
    elif file_type == "pdf":
        df = read_pdf(uploaded_file)
    else:
        df = None

    if df is not None:
        st.success("✅ File uploaded successfully!")
        st.markdown("### 👁 Preview of Data")
        st.dataframe(df.head(10), use_container_width=True)

        st.sidebar.markdown("---")
        mode = st.sidebar.radio("Choose an Analysis Mode:", [
            "📈 Lead Scoring",
            "🧠 Customer Segmentation",
            "📊 Trend Forecasting",
            "💀 Churn Analysis",
            "💬 Chat with Data"
        ])

        st.markdown("---")
        result_df = None

        if mode == "📈 Lead Scoring":
            st.subheader("📈 Lead Scoring")
            if all(col in df.columns for col in ['monthly_spend', 'support_tickets']):
                df['lead_score'] = (
                    0.4 * df['monthly_spend'].fillna(0) +
                    0.3 * df['support_tickets'].fillna(0)
                )
                top_leads = df.sort_values(by='lead_score', ascending=False).head(10)
                st.dataframe(top_leads, use_container_width=True)
                result_df = top_leads
            else:
                st.warning("Required columns not found: 'monthly_spend', 'support_tickets'")

        elif mode == "🧠 Customer Segmentation":
            st.subheader("🧠 Customer Segmentation")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) >= 2:
                k = st.slider("Number of segments:", 2, 6, 3)
                kmeans = KMeans(n_clusters=k, random_state=42)
                df['segment'] = kmeans.fit_predict(df[numeric_cols].fillna(0))
                st.dataframe(df[['segment'] + numeric_cols[:3]], use_container_width=True)
                st.bar_chart(df['segment'].value_counts())
                result_df = df[['segment'] + numeric_cols[:3]]
            else:
                st.warning("Not enough numeric columns for clustering.")

        elif mode == "📊 Trend Forecasting":
            st.subheader("📊 Trend Forecasting")
            date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            if date_cols:
                date_col = st.selectbox("Select a date column", date_cols)
                metric = st.selectbox("Metric column", df.select_dtypes(include='number').columns)
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                trend = df[[date_col, metric]].dropna().groupby(date_col)[metric].mean()
                st.line_chart(trend)
            else:
                st.warning("No valid date/time column found.")

        elif mode == "💀 Churn Analysis":
            st.subheader("💀 Churn Analysis")
            if 'churned' in df.columns:
                churn_rate = df['churned'].mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.2f}%")
                st.bar_chart(df['churned'].value_counts())
            else:
                st.warning("'churned' column not found.")

        elif mode == "💬 Chat with Data":
            st.subheader("💬 Ask Questions About Your Data")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for i, (q, a, df_resp) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    st.markdown(a)
                    if df_resp is not None:
                        st.dataframe(df_resp, use_container_width=True)

            user_input = st.chat_input("Ask something about your data...")
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    response = analyze_question(df, user_input)
                    st.markdown(response[0])
                    if response[1] is not None:
                        st.dataframe(response[1], use_container_width=True)
                        # Show chart if numeric column mentioned
                        for col in df.select_dtypes(include='number').columns:
                            if col.lower() in user_input.lower():
                                render_chart_in_chat(df, col)
                                break
                    st.session_state.chat_history.append((user_input, response[0], response[1]))

        if result_df is not None:
            st.markdown("---")
            if st.button("📄 Export Result to PDF"):
                pdf_data = convert_df_to_pdf(result_df)
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name="terraga_result.pdf",
                    mime="application/pdf"
                )
else:
    st.info("📂 Please upload a file to begin.")