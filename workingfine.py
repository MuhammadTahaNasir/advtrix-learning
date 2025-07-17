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
from transformers import pipeline
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =====================
# Terrag AI - Market Assistance
# =====================
st.title("📊 Terrag AI - Market Assistance")
st.subheader("Upload a CSV, Excel, or PDF file to analyze your data with AI-powered insights!")
st.markdown("**Get started**: Upload a file, choose an analysis mode, and ask questions or explore insights. Everything is free!")

# How to Use Section
with st.expander("ℹ️ How to Use This App", expanded=True):
    st.markdown("""
    **Welcome to Terrag AI!** This app helps you analyze business data with smartly. Here's how it works:
    - **Upload a File**: Use the sidebar to upload a CSV, Excel, or PDF file with your data (e.g., customer sales, feedback).
    - **Choose a Mode**: Pick from Lead Scoring, Customer Segmentation, Trend Forecasting, Churn Analysis, Sentiment Analysis, or Chat with Data.
    - **Explore Insights**: View automated insights, filter data, or ask questions like "What’s the average sales?" or "Drop rows where revenue < 100".
    - **Export Results**: Download your analysis as a PDF.
    - **Example Questions**: Try "Show me the top monthly_spend" or "Which customers are in segment 1?" in Chat mode.
    """)

st.sidebar.header("📁 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV, Excel, or PDF file", type=["csv", "xlsx", "xls", "pdf"])

# ---------- Model Loading with Caching ----------
@st.cache_resource(show_spinner=True)
def load_sentence_model():
    model_path = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        return SentenceTransformer(model_path if os.path.exists(model_path) else 'all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load sentence model: {str(e)}. Some features may be limited.")
        return None

@st.cache_resource(show_spinner=True)
def load_qa_model():
    model_path = os.path.join(os.getcwd(), "models", "flan-t5-base")
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        return pipeline("text2text-generation", model=model_path if os.path.exists(model_path) else "google/flan-t5-base")
    except Exception as e:
        st.error(f"Failed to load QA model: {str(e)}. Some features may be limited.")
        return None

@st.cache_resource(show_spinner=True)
def load_sentiment_model():
    model_path = os.path.join(os.getcwd(), "models", "distilbert-base-uncased-finetuned-sst-2-english")
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        return pipeline("sentiment-analysis", model=model_path if os.path.exists(model_path) else "distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Failed to load sentiment model: {str(e)}. Some features may be limited.")
        return None

sentence_model = load_sentence_model()
qa_model = load_qa_model()
sentiment_model = load_sentiment_model()

# ---------- PDF Reader Helper ----------
def read_pdf(file):
    try:
        import fitz  # PyMuPDF
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
        return pd.DataFrame({"Text": text.split("\n")})
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}. Please ensure it’s a table-based PDF.")
        return None

# ---------- Smart Q&A Function with Conversational Memory ----------
def analyze_question(df, question, context=""):
    lower_q = question.lower()
    prompt = f"Context: {context}\nData Summary: {df.describe().to_string()}\nQuestion: {question}\nAnswer clearly and concisely:"
    
    # Direct column-based answers
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
            elif any(k in lower_q for k in ["sum", "total"]):
                try:
                    total_val = df[col].sum()
                    return (f"📈 The total `{col}` is **{total_val:.2f}**.", None)
                except:
                    continue

    if sentence_model is None:
        return ("⚠️ Semantic search unavailable due to model loading issue.", None)

    # Semantic search for relevant rows
    row_texts = df.astype(str).agg(" ".join, axis=1).tolist()
    row_embeddings = sentence_model.encode(row_texts, show_progress_bar=False)
    question_embedding = sentence_model.encode([question])
    sims = cosine_similarity(question_embedding, row_embeddings)[0]
    top_index = sims.argmax()
    top_row = df.iloc[[top_index]]
    
    # Generate precise answer
    context_summary = " ".join(row_texts[:3])  # First few rows as context
    answer = qa_model(prompt, max_length=150, num_beams=5)[0]['generated_text'] if qa_model else "⚠️ QA model unavailable."
    return (f"💡 {answer}", top_row)

# ---------- Sentiment Analysis ----------
def analyze_sentiment(df, text_col):
    if sentiment_model is None:
        st.warning("⚠️ Sentiment analysis unavailable due to model loading issue.")
        return df
    try:
        sentiments = sentiment_model(df[text_col].astype(str).tolist())
        df['sentiment'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        return df
    except Exception as e:
        st.warning(f"Error in sentiment analysis: {str(e)}")
        return df

# ---------- Data Cleaning Suggestions ----------
def suggest_data_cleaning(df):
    suggestions = []
    if df.isnull().sum().sum() > 0:
        suggestions.append(f"Found {df.isnull().sum().sum()} missing values. Suggested fix: Fill with mean/median or drop rows.")
    if df.duplicated().sum() > 0:
        suggestions.append(f"Found {df.duplicated().sum()} duplicate rows. Suggested fix: Remove duplicates.")
    return "\n".join(suggestions) or "✅ No major issues detected."

# ---------- Automated Insights ----------
def generate_insights(df):
    insights = []
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        mean_val = df[col].mean()
        max_val = df[col].max()
        min_val = df[col].min()
        insights.append(f"📊 `{col}`: Mean = {mean_val:.2f}, Max = {max_val}, Min = {min_val}")
    summary = "\n".join(insights)
    prompt = f"Summarize these insights in a concise paragraph:\n{summary}"
    return qa_model(prompt, max_length=200, num_beams=5)[0]['generated_text'] if qa_model else "⚠️ Insights unavailable due to model loading issue."

# ---------- Segment Descriptions ----------
def describe_segments(df, segment_col, numeric_cols):
    descriptions = []
    for segment in df[segment_col].unique():
        segment_data = df[df[segment_col] == segment][numeric_cols]
        stats = segment_data.describe().loc[['mean', 'min', 'max']].to_dict()
        description = f"Segment {segment}: "
        for col in stats:
            description += f"{col} (Mean: {stats[col]['mean']:.2f}, Range: {stats[col]['min']:.2f}-{stats[col]['max']:.2f}), "
        descriptions.append(description)
    return "\n".join(descriptions)

# ---------- Recommendation Engine ----------
def recommend_actions(df):
    recommendations = []
    if 'lead_score' in df.columns:
        high_leads = df[df['lead_score'] > df['lead_score'].quantile(0.8)]
        recommendations.append(f"🎯 Target {len(high_leads)} high-value leads with scores above {df['lead_score'].quantile(0.8):.2f}.")
    if 'segment' in df.columns:
        top_segment = df['segment'].value_counts().idxmax()
        recommendations.append(f"📈 Focus marketing on Segment {top_segment}, which has {df['segment'].value_counts().max()} customers.")
    if 'sentiment' in df.columns:
        positive_count = len(df[df['sentiment'] == 'POSITIVE'])
        recommendations.append(f"😊 Leverage {positive_count} positive customer sentiments for testimonials.")
    return "\n".join(recommendations) or "No specific recommendations available."

# ---------- Natural Language Data Manipulation ----------
def parse_data_command(command, df):
    try:
        if "drop rows" in command.lower() and "where" in command.lower():
            parts = command.lower().split("where")
            condition = parts[1].strip()
            if "<" in condition:
                col, value = condition.split("<")
                col, value = col.strip(), float(value.strip())
                return df[df[col] >= value]
            elif ">" in condition:
                col, value = condition.split(">")
                col, value = col.strip(), float(value.strip())
                return df[df[col] <= value]
        return df
    except Exception as e:
        st.warning(f"Error parsing command: {str(e)}")
        return df

# ---------- PDF Export Helper ----------
def convert_df_to_pdf(df):
    try:
        fig, ax = plt.subplots(figsize=(12, min(20, len(df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        buf = BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# ---------- Chart in Chat ----------
def render_chart_in_chat(df, col):
    try:
        fig, ax = plt.subplots()
        df[col].value_counts().head(10).plot(kind='bar', ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Error rendering chart: {str(e)}")

# ---------- App Logic ----------
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_type == "pdf":
            df = read_pdf(uploaded_file)
        else:
            df = None
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        df = None

    if df is not None:
        # Downsample for performance
        if len(df) > 10000:
            st.warning("⚠️ Large dataset detected. Downsampling to 10,000 rows for faster processing.")
            df = df.sample(10000, random_state=42)

        st.success("✅ File uploaded successfully!")
        st.markdown("### 👁 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Data Cleaning Suggestions
        with st.expander("🧹 Data Cleaning Suggestions", expanded=True):
            st.markdown(suggest_data_cleaning(df))
            if st.button("Apply Cleaning Fixes"):
                df = df.fillna(df.mean(numeric_only=True)).drop_duplicates()
                st.dataframe(df.head(10))

        # Automated Insights
        with st.expander("🔍 Automated Insights", expanded=True):
            st.markdown(generate_insights(df))

            numeric_cols = df.select_dtypes(include='number').columns

            if len(numeric_cols) > 0:
                st.markdown("**Key Metrics Visualization**")

                import altair as alt
                import pandas as pd

                # Prepare data for bar chart
                metrics_df = pd.DataFrame({
                    'Column': numeric_cols,
                    'Mean Value': [df[col].mean() for col in numeric_cols]
                })

                bar_chart = alt.Chart(metrics_df).mark_bar().encode(
                    x=alt.X('Column', sort=None, title='Numeric Columns'),
                    y=alt.Y('Mean Value', title='Mean'),
                    tooltip=['Column', 'Mean Value']
                ).properties(
                    width=700,
                    height=400,
                    title="Mean Values of Numeric Columns"
                ).configure_mark(
                    color='#4CAF50'
                )

                st.altair_chart(bar_chart, use_container_width=True)

                # Interactive Filters
                st.sidebar.markdown("### 🔍 Data Filters")
                filtered_df = df.copy()
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        min_val, max_val = df[col].min(), df[col].max()
                        selected_range = st.sidebar.slider(f"Filter {col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
                        filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]
                with st.expander("📋 Filtered Data", expanded=False):
                    st.dataframe(filtered_df, use_container_width=True)

                st.sidebar.markdown("---")
                mode = st.sidebar.radio("Choose an Analysis Mode:", [
                    "📈 Lead Scoring",
                    "🧠 Customer Segmentation",
                    "📊 Trend Forecasting",
                    "💀 Churn Analysis",
                    "😊 Sentiment Analysis",
                    "💬 Chat with Data"
                ])

                st.markdown("---")
                result_df = None

                if mode == "📈 Lead Scoring":
                    with st.expander("📈 Lead Scoring", expanded=True):
                        st.markdown("**What it does**: Scores leads based on monthly_spend and support_tickets to identify high-value customers.")
                        if all(col in df.columns for col in ['monthly_spend', 'support_tickets']):
                            df['lead_score'] = (
                                0.4 * df['monthly_spend'].fillna(0) +
                                0.3 * df['support_tickets'].fillna(0)
                            )
                            top_leads = df.sort_values(by='lead_score', ascending=False).head(10)
                            st.dataframe(top_leads, use_container_width=True)
                            result_df = top_leads
                        else:
                            st.warning("⚠️ Required columns not found: 'monthly_spend', 'support_tickets'")

                elif mode == "🧠 Customer Segmentation":
                    with st.expander("🧠 Customer Segmentation", expanded=True):
                        st.markdown("**What it does**: Groups customers into segments based on numeric data for targeted marketing.")
                        numeric_cols = df.select_dtypes(include='number').columns.tolist()
                        if len(numeric_cols) >= 2:
                            k = st.slider("Number of segments:", 2, 6, 3)
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            df['segment'] = kmeans.fit_predict(df[numeric_cols].fillna(0))
                            st.dataframe(df[['segment'] + numeric_cols[:3]], use_container_width=True)
                            st.markdown("### Segment Descriptions")
                            st.markdown(describe_segments(df, 'segment', numeric_cols[:3]))
                            st.bar_chart(df['segment'].value_counts())
                            # Scatter plot for first 2 numeric features
                            if len(numeric_cols) >= 2:
                                import altair as alt
                                chart_data = df[[numeric_cols[0], numeric_cols[1], 'segment']].copy()
                                chart_data['segment'] = chart_data['segment'].astype(str)
                                scatter = alt.Chart(chart_data).mark_circle(size=60).encode(
                                    x=alt.X(numeric_cols[0], title=numeric_cols[0]),
                                    y=alt.Y(numeric_cols[1], title=numeric_cols[1]),
                                    color='segment',
                                    tooltip=[numeric_cols[0], numeric_cols[1], 'segment']
                                ).properties(
                                    width=600,
                                    height=400,
                                    title='Customer Segments (2D Scatter Plot)'
                                )
                                st.altair_chart(scatter, use_container_width=True)
                            result_df = df[['segment'] + numeric_cols[:3]]
                        else:
                            st.warning("⚠️ Not enough numeric columns for clustering.")

                elif mode == "📊 Trend Forecasting":
                    with st.expander("📊 Trend Forecasting", expanded=True):
                        st.markdown("**What it does**: Shows trends over time for a selected metric (e.g., sales over dates).")
                        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
                        if date_cols:
                            date_col = st.selectbox("Select a date column", date_cols)
                            metric = st.selectbox("Metric column", df.select_dtypes(include='number').columns)
                            try:
                                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                                trend = df[[date_col, metric]].dropna().groupby(date_col)[metric].mean()
                                st.line_chart(trend)
                            except Exception as e:
                                st.warning(f"⚠️ Error in trend forecasting: {str(e)}")
                        else:
                            st.warning("⚠️ No valid date/time column found.")

                elif mode == "💀 Churn Analysis":
                    with st.expander("💀 Churn Analysis", expanded=True):
                        st.markdown("**What it does**: Analyzes customer churn rates if a 'churned' column exists.")
                        if 'churned' in df.columns:
                            churn_rate = df['churned'].mean() * 100
                            st.metric("Churn Rate", f"{churn_rate:.2f}%")
                            st.bar_chart(df['churned'].value_counts())
                        else:
                            st.warning("⚠️ 'churned' column not found.")

                elif mode == "😊 Sentiment Analysis":
                    with st.expander("😊 Sentiment Analysis", expanded=True):
                        st.markdown("**What it does**: Analyzes text (e.g., customer feedback) to determine positive or negative sentiment.")
                        text_cols = df.select_dtypes(include='object').columns
                        if len(text_cols) > 0:  # Fixed: Check length of text_cols
                            selected_text_col = st.selectbox("Select a text column", text_cols)
                            df = analyze_sentiment(df, selected_text_col)
                            st.dataframe(df[[selected_text_col, 'sentiment', 'sentiment_score']], use_container_width=True)
                            st.bar_chart(df['sentiment'].value_counts())
                            result_df = df[[selected_text_col, 'sentiment', 'sentiment_score']]
                        else:
                            st.warning("⚠️ No text columns found for sentiment analysis.")

                elif mode == "💬 Chat with Data":
                    with st.expander("💬 Chat with Data", expanded=True):
                        st.markdown("**What it does**: Ask questions about your data (e.g., 'What’s the top sales?') or give commands (e.g., 'Drop rows where monthly_spend < 100').")
                        if "chat_history" not in st.session_state:
                            st.session_state.chat_history = []
                        if "chat_context" not in st.session_state:
                            st.session_state.chat_context = ""
                        for i, (q, a, df_resp) in enumerate(st.session_state.chat_history):
                            with st.chat_message("user"):
                                st.markdown(q)
                            with st.chat_message("assistant"):
                                st.markdown(a)
                                if df_resp is not None:
                                    st.dataframe(df_resp, use_container_width=True)
                        user_input = st.chat_input("Ask a question or give a data command...")  # Fixed: Removed help parameter
                        if user_input:
                            with st.chat_message("user"):
                                st.markdown(user_input)
                            with st.chat_message("assistant"):
                                if "drop rows" in user_input.lower():
                                    df = parse_data_command(user_input, df)
                                    st.markdown("✅ Data updated based on your command.")
                                    st.dataframe(df.head(10))
                                else:
                                    response = analyze_question(df, user_input, st.session_state.chat_context)
                                    st.markdown(response[0])
                                    if response[1] is not None:
                                        st.dataframe(response[1], use_container_width=True)
                                        for col in df.select_dtypes(include='number').columns:
                                            if col.lower() in user_input.lower():
                                                render_chart_in_chat(df, col)
                                                break
                                    st.session_state.chat_history.append((user_input, response[0], response[1]))
                                    st.session_state.chat_context += f"User: {user_input}\nAssistant: {response[0]}\n"

                # Recommendations
                with st.expander("🎯 Action Recommendations", expanded=True):
                    st.markdown("**What it does**: Suggests actions based on your data, like targeting high-value leads.")
                    st.markdown(recommend_actions(df))

                # Export Result
                if result_df is not None:
                    st.markdown("---")
                    if st.button("Export Result to PDF"):
                        pdf_data = convert_df_to_pdf(result_df)
                        if pdf_data:
                            st.download_button(
                                label="Download PDF",
                                data=pdf_data,
                                file_name="terraga_result.pdf",
                                mime="application/pdf"
                            )

                # Help Section
                st.sidebar.markdown("---")
                if st.sidebar.button("Explain Features"):
                    prompt = "Explain lead scoring, customer segmentation, trend forecasting, churn analysis, and sentiment analysis in simple terms."
                    explanation = qa_model(prompt, max_length=300, num_beams=5)[0]['generated_text'] if qa_model else "⚠️ Explanation unavailable due to model loading issue."
                    with st.expander("ℹ️ Feature Explanations", expanded=True):
                        st.markdown(explanation)
else:
    st.info("📂 Please upload a file to begin. Supported formats: CSV, Excel, PDF.")