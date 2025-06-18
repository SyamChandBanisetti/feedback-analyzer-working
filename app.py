import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from collections import Counter
import io
import os
from dotenv import load_dotenv
import time
import uuid

# Load .env and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Cache model
@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

# Gemini-based sentiment classification
def analyze_sentiments_with_gemini(texts):
    model = get_gemini_model()
    all_results = []

    for text in texts:
        text = text.strip()
        if not text:
            all_results.append(("NEUTRAL", "Empty response"))
            continue

        text = text[:400]  # Truncate if needed

        prompt = f"""
You are a sentiment analysis expert. Classify the sentiment of the following text as POSITIVE, NEGATIVE, or NEUTRAL and provide a brief reason.

Text: "{text}"

Return format: <label> - <reason>
"""

        try:
            response = model.generate_content(prompt)
            content = response.text.strip()

            if "-" in content:
                label, reason = content.split("-", 1)
                label = label.strip().upper()
                reason = reason.strip()
                if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                    label, reason = "NEUTRAL", "Unrecognized sentiment"
            else:
                label, reason = "NEUTRAL", "Invalid response format"

        except Exception as e:
            label, reason = "NEUTRAL", "API Error"

        all_results.append((label, reason))

    return all_results

# Analyze each column
def analyze_sentiment_distribution(series, column_name):
    texts = series.dropna().astype(str).tolist()

    if not texts:
        return None

    results = analyze_sentiments_with_gemini(texts)
    labels = [res[0] for res in results]
    reasons = [res[1] for res in results]

    counts = Counter(labels)
    total = len(results)

    percent = {
        "Positive": round((counts.get("POSITIVE", 0) / total) * 100, 1),
        "Negative": round((counts.get("NEGATIVE", 0) / total) * 100, 1),
        "Neutral": round((counts.get("NEUTRAL", 0) / total) * 100, 1)
    }

    # Summarization prompt
    summary_prompt = f"""
Analyze this sentiment breakdown for the question: "{column_name}" — {percent}.
Give a short:
1. Summary
2. Insight
3. Practical recommendation

Return format:
Summary: ...
Insights: ...
Recommendations: ...
"""

    try:
        response = get_gemini_model().generate_content(summary_prompt)
        response_text = response.text.strip()
    except Exception:
        response_text = "Summary: Not available\nInsights: API limit reached\nRecommendations: Try again later"

    summary = insights = recommendations = ""
    for line in response_text.splitlines():
        if line.startswith("Summary:"):
            summary = line.replace("Summary:", "").strip()
        elif line.startswith("Insights:"):
            insights = line.replace("Insights:", "").strip()
        elif line.startswith("Recommendations:"):
            recommendations = line.replace("Recommendations:", "").strip()

    return {
        "Total": total,
        "Positive": counts.get("POSITIVE", 0),
        "Negative": counts.get("NEGATIVE", 0),
        "Neutral": counts.get("NEUTRAL", 0),
        "Percentages": percent,
        "Summary": summary,
        "Insights": insights,
        "Recommendations": recommendations,
        "Details": list(zip(texts, labels, reasons))
    }

# Streamlit UI
st.set_page_config(page_title="📊 Feedback Sentiment Analyzer", layout="wide")
st.title("🧠 Gemini-Powered Feedback Analyzer")
st.markdown("Upload a feedback CSV file to get automated sentiment analysis and insights.")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_columns = df.select_dtypes(include="object").columns.tolist()

    ignore = ["timestamp", "email", "id", "name"]
    text_columns = [col for col in text_columns if col.lower() not in ignore]

    if not text_columns:
        st.warning("No text-based columns detected.")
    else:
        summary_data = []

        for idx, col in enumerate(text_columns):
            st.markdown(f"---\n### 📌 Question {idx+1}: **{col}**")

            result = analyze_sentiment_distribution(df[col], col)
            if not result:
                st.warning("No valid responses found.")
                continue

            col1, col2 = st.columns([1.5, 2])

            sentiment_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Count": [result["Positive"], result["Negative"], result["Neutral"]]
            })

            unique_id = str(uuid.uuid4())[:8]  # ensure unique chart ID

            with col1:
                pie = px.pie(sentiment_data, values="Count", names="Sentiment", title="Sentiment Distribution")
                pie.update_traces(textinfo="percent+label")
                st.plotly_chart(pie, use_container_width=True, key=f"pie-{unique_id}")

            with col2:
                st.metric("🧾 Total", result["Total"])
                st.metric("✅ Positive", f"{result['Positive']} ({result['Percentages']['Positive']}%)")
                st.metric("❌ Negative", f"{result['Negative']} ({result['Percentages']['Negative']}%)")
                st.metric("➖ Neutral", f"{result['Neutral']} ({result['Percentages']['Neutral']}%)")

                bar = px.bar(sentiment_data, x="Sentiment", y="Count", color="Sentiment", text="Count")
                st.plotly_chart(bar, use_container_width=True, key=f"bar-{unique_id}")

            st.markdown(f"**📝 Summary**: {result['Summary']}")
            st.markdown(f"**🔎 Insights**: {result['Insights']}")
            st.markdown(f"**✅ Recommendations**: {result['Recommendations']}")

            with st.expander("🔍 Sample Responses & Reasoning"):
                detail_df = pd.DataFrame(result["Details"], columns=["Response", "Sentiment", "Reason"])
                st.dataframe(detail_df.head(10), use_container_width=True)

            summary_data.append({
                "Question": col,
                "Total": result["Total"],
                "Positive %": result["Percentages"]["Positive"],
                "Negative %": result["Percentages"]["Negative"],
                "Neutral %": result["Percentages"]["Neutral"],
                "Summary": result["Summary"],
                "Insights": result["Insights"],
                "Recommendations": result["Recommendations"]
            })

        # Final report
        if summary_data:
            st.markdown("### 📥 Download Overall Report")
            summary_df = pd.DataFrame(summary_data)
            buffer = io.BytesIO()
            summary_df.to_excel(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="📊 Download Excel Summary",
                data=buffer,
                file_name="sentiment_feedback_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
