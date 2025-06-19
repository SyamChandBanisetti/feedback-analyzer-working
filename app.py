import streamlit as st
import os
import re
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env and fallback to Streamlit secrets
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("gemini", {}).get("api_key", "")

# Configure page
st.set_page_config(page_title="📊 Feedback Summarizer", layout="wide")

# --- CSV parsing and cleaning ---
def parse_csv_simple(file_content):
    lines = file_content.strip().split('\n')
    if not lines:
        return [], []
    header = [col.strip().strip('"') for col in lines[0].split(',')]
    data = []
    for line in lines[1:]:
        if line.strip():
            row = [cell.strip().strip('"') for cell in line.split(',')]
            if len(row) < len(header):
                row += [''] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[:len(header)]
            data.append(row)
    return header, data

def clean_text(text):
    return re.sub(r'\s+', ' ', str(text).strip()) if text and str(text).strip() else ''

def is_numeric_response(responses):
    numeric_count = 0
    total_count = 0
    for r in responses:
        r = r.strip()
        if r:
            total_count += 1
            try:
                float(r)
                numeric_count += 1
            except:
                pass
    return total_count > 0 and numeric_count / total_count > 0.7

# --- Basic fallback analysis ---
def analyze_question_basic(question, responses):
    from collections import Counter
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    keywords = []
    for r in responses:
        t = r.lower()
        if any(w in t for w in ["good", "great", "love", "helpful", "excellent"]):
            sentiments["Positive"] += 1
        elif any(w in t for w in ["bad", "poor", "worst", "problem", "issue", "slow"]):
            sentiments["Negative"] += 1
        else:
            sentiments["Neutral"] += 1
    all_words = ' '.join(responses).lower().split()
    common = Counter(all_words)
    keywords = [w for w, _ in common.most_common(5) if len(w) > 3]
    summary = f"Based on {len(responses)} responses, feedback is mostly {max(sentiments, key=sentiments.get).lower()}."
    return {
        "summary": summary,
        "response_count": len(responses),
        "avg_rating": None,
        "sentiment": sentiments,
        "keywords": keywords
    }

# --- Gemini-powered AI analysis ---
def analyze_question_with_ai(question, responses, api_key):
    valid = [clean_text(r) for r in responses if clean_text(r)]
    if not valid:
        return {"summary": "No meaningful responses.", "response_count": 0, "avg_rating": None, "sentiment": {}, "keywords": []}
    
    if is_numeric_response(valid):
        nums = [float(r) for r in valid if r.replace('.', '', 1).isdigit()]
        avg = sum(nums) / len(nums)
        return {
            "summary": f"Average rating: {avg:.2f} from {len(nums)} responses.",
            "response_count": len(nums),
            "avg_rating": avg,
            "sentiment": {},
            "keywords": []
        }

    if not api_key:
        return analyze_question_basic(question, valid)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        sample = valid[:15]
        prompt = f"""
Analyze the following feedback for the question: "{question}".

Responses:
{chr(10).join(['- ' + r for r in sample])}

Write:
SUMMARY:
[2-3 sentence professional-style summary of key insights, tone, and suggestions.]

SENTIMENT:
Positive: X
Negative: Y
Neutral: Z

KEYWORDS:
word1, word2, word3, word4, word5
"""
        response = model.generate_content(prompt)
        text = response.text
        time.sleep(2)

        # Parse result
        summary = ""
        sentiment = {}
        keywords = []
        for block in text.split('\n\n'):
            if block.startswith("SUMMARY:"):
                summary = block.replace("SUMMARY:", "").strip()
            elif block.startswith("SENTIMENT:"):
                lines = block.splitlines()[1:]
                for l in lines:
                    if ':' in l:
                        k, v = l.split(':')
                        sentiment[k.strip()] = int(v.strip())
            elif block.startswith("KEYWORDS:"):
                kwline = block.replace("KEYWORDS:", "").strip()
                keywords = [k.strip() for k in kwline.split(',')]
        return {
            "summary": summary or f"Analysis of {len(valid)} responses shows various themes.",
            "response_count": len(valid),
            "avg_rating": None,
            "sentiment": sentiment,
            "keywords": keywords
        }
    except Exception as e:
        st.warning(f"AI error: {e}. Using basic analysis.")
        return analyze_question_basic(question, valid)

# --- Main Streamlit UI ---
def main():
    st.title("📊 Feedback Summarizer")
    st.markdown("Analyze and summarize responses using AI (or basic fallback).")

    # Upload file
    with st.sidebar:
        st.header("Upload CSV")
        uploaded = st.file_uploader("Upload", type=["csv"])
        if GEMINI_API_KEY:
            st.success("✅ AI mode enabled")
        else:
            st.warning("⚠️ Using fallback mode (no AI key found)")

    if not uploaded:
        st.info("Upload a feedback CSV to begin.")
        return

    content = uploaded.getvalue().decode("utf-8")
    headers, data = parse_csv_simple(content)

    if not headers or not data:
        st.error("Invalid file.")
        return

    st.success(f"Loaded {len(data)} responses and {len(headers)} questions.")
    df_display = pd.DataFrame(data, columns=headers)
    st.dataframe(df_display.head())

    st.markdown("## 📝 Question Analysis")
    report_lines = [
        "FEEDBACK SUMMARY REPORT",
        "="*40,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Responses: {len(data)}",
        f"Questions: {len(headers)}",
        ""
    ]

    for i, question in enumerate(headers):
        responses = [row[i] if i < len(row) else '' for row in data]
        analysis = analyze_question_with_ai(question, responses, GEMINI_API_KEY)

        with st.expander(f"Q{i+1}: {question}", expanded=False):
            st.markdown(f"**Summary:**\n\n{analysis['summary']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Responses", analysis['response_count'])
            if analysis['avg_rating']:
                c2.metric("Avg Rating", f"{analysis['avg_rating']:.2f}")
            else:
                c2.metric("Response Type", "Text")
            if len(data) > 0:
                c3.metric("Response Rate", f"{(analysis['response_count']/len(data))*100:.1f}%")

            if analysis["sentiment"]:
                st.markdown("**Sentiment:**")
                s1, s2, s3 = st.columns(3)
                s1.metric("Positive", analysis["sentiment"].get("Positive", 0))
                s2.metric("Neutral", analysis["sentiment"].get("Neutral", 0))
                s3.metric("Negative", analysis["sentiment"].get("Negative", 0))

            if analysis["keywords"]:
                st.markdown("**Top Keywords:**")
                st.write("• " + " • ".join(analysis["keywords"]))

        # Build report line
        report_lines.append(f"Q{i+1}: {question}")
        report_lines.append("-" * 30)
        report_lines.append("Summary: " + analysis["summary"])
        if analysis["sentiment"]:
            report_lines.append("Sentiment: " + str(analysis["sentiment"]))
        if analysis["keywords"]:
            report_lines.append("Keywords: " + ", ".join(analysis["keywords"]))
        report_lines.append("")

    # Download section
    st.markdown("## 📥 Download Summary")
    st.download_button(
        "📄 Download .txt Report",
        data="\n".join(report_lines),
        file_name=f"feedback_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
