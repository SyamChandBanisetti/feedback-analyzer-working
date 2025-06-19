import streamlit as st
import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configure page
st.set_page_config(
    page_title="Feedback Form Summarizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_csv_simple(file_content):
    """Simple CSV parser"""
    lines = file_content.strip().split('\n')
    if not lines:
        return [], []
    
    # Parse header
    header = [col.strip().strip('"') for col in lines[0].split(',')]
    
    # Parse data rows
    data = []
    for line in lines[1:]:
        if line.strip():
            row = [cell.strip().strip('"') for cell in line.split(',')]
            # Pad row if shorter than header
            while len(row) < len(header):
                row.append('')
            data.append(row)
    
    return header, data

def clean_text(text):
    """Clean and normalize text data"""
    if not text or text.strip() == '':
        return ''
    
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_numeric_response(responses):
    """Check if responses are mostly numeric (ratings)"""
    if not responses:
        return False
    
    numeric_count = 0
    total_count = 0
    
    for response in responses:
        if response and response.strip():
            total_count += 1
            try:
                float(response)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
    
    return total_count > 0 and (numeric_count / total_count) > 0.7

def analyze_question_with_ai(question, responses, api_key):
    """Analyze a single question using Gemini AI"""
    if not responses:
        return {
            'summary': 'No responses available for this question.',
            'response_count': 0,
            'avg_rating': None,
            'sentiment': {},
            'keywords': []
        }
    
    # Filter out empty responses
    valid_responses = [clean_text(r) for r in responses if clean_text(r)]
    
    if not valid_responses:
        return {
            'summary': 'No meaningful responses found for this question.',
            'response_count': 0,
            'avg_rating': None,
            'sentiment': {},
            'keywords': []
        }
    
    # Check if this is a rating question
    if is_numeric_response(valid_responses):
        # Handle numeric responses
        numeric_values = []
        for response in valid_responses:
            try:
                numeric_values.append(float(response))
            except:
                continue
        
        if numeric_values:
            avg_rating = sum(numeric_values) / len(numeric_values)
            max_rating = max(numeric_values)
            
            return {
                'summary': f"Average rating: {avg_rating:.2f} out of {max_rating:.0f} based on {len(numeric_values)} responses. Rating distribution shows {len([r for r in numeric_values if r >= avg_rating])} responses at or above average.",
                'response_count': len(numeric_values),
                'avg_rating': avg_rating,
                'sentiment': {},
                'keywords': []
            }
    
    # Handle text responses with AI
    if not api_key:
        return analyze_question_basic(question, valid_responses)
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Limit responses for API efficiency
        sample_responses = valid_responses[:15] if len(valid_responses) > 15 else valid_responses
        
        prompt = f"""
        Analyze the following feedback responses for the question: "{question}"

        Responses:
        {chr(10).join([f"- {response}" for response in sample_responses])}

        Please provide a natural, comprehensive analysis that tells the story of what students/respondents are saying. Write it like a professional summary report that highlights key themes, common issues, and overall sentiment in a narrative format.

        Example style: "Students appreciated the clarity of the instructor but suggested a slower pace and more real-world examples in lectures. The majority found the content engaging, though several mentioned difficulty keeping up with the fast-paced delivery."

        Also provide:
        1. Sentiment counts (format: Positive: X, Negative: Y, Neutral: Z)
        2. Top 5 keywords or themes mentioned

        Format your response exactly like this:
        
        SUMMARY:
        [Write a natural, story-like summary that captures the essence of the feedback - 2-3 sentences that flow naturally and highlight key insights, common themes, and overall sentiment]
        
        SENTIMENT:
        Positive: X
        Negative: Y
        Neutral: Z
        
        KEYWORDS:
        keyword1, keyword2, keyword3, keyword4, keyword5
        """
        
        response = model.generate_content(prompt)
        ai_response = response.text
        
        # Add delay to respect rate limits
        time.sleep(2)
        
        # Parse the AI response
        summary = ""
        sentiment = {}
        keywords = []
        
        sections = ai_response.split('\n\n')
        for section in sections:
            if section.startswith('SUMMARY:'):
                summary = section.replace('SUMMARY:', '').strip()
            elif section.startswith('SENTIMENT:'):
                sentiment_text = section.replace('SENTIMENT:', '').strip()
                for line in sentiment_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            sentiment[key.strip()] = int(value.strip())
                        except:
                            pass
            elif section.startswith('KEYWORDS:'):
                keywords_text = section.replace('KEYWORDS:', '').strip()
                keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        
        return {
            'summary': summary or f"Analysis of {len(valid_responses)} responses shows various feedback themes.",
            'response_count': len(valid_responses),
            'avg_rating': None,
            'sentiment': sentiment,
            'keywords': keywords
        }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            st.warning("⏳ Rate limit reached. Waiting before continuing with basic analysis...")
            time.sleep(5)
            return analyze_question_basic(question, valid_responses)
        else:
            st.warning(f"AI analysis failed: {error_msg}. Using basic analysis.")
            return analyze_question_basic(question, valid_responses)

def analyze_question_basic(question, valid_responses):
    """Basic analysis without AI"""
    from utils import simple_sentiment_analysis, extract_keywords_simple
    
    sentiment = simple_sentiment_analysis(valid_responses)
    keywords = extract_keywords_simple(valid_responses)
    
    # Create narrative summary
    response_count = len(valid_responses)
    
    # Analyze sentiment distribution
    positive_count = sentiment.get('Positive', 0)
    negative_count = sentiment.get('Negative', 0)
    neutral_count = sentiment.get('Neutral', 0)
    
    # Determine overall sentiment
    if positive_count > negative_count and positive_count > neutral_count:
        overall_sentiment = "mostly positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        overall_sentiment = "mostly negative"
    else:
        overall_sentiment = "mixed"
    
    # Create a narrative summary
    summary = f"Based on {response_count} responses, feedback is {overall_sentiment}."
    
    if keywords:
        top_themes = keywords[:3]
        if len(top_themes) == 1:
            summary += f" The main theme mentioned was '{top_themes[0]}'."
        elif len(top_themes) == 2:
            summary += f" Key themes include '{top_themes[0]}' and '{top_themes[1]}'."
        else:
            summary += f" Common themes mentioned were '{top_themes[0]}', '{top_themes[1]}', and '{top_themes[2]}'."
    
    # Add sentiment context
    if positive_count > 0 and negative_count > 0:
        summary += f" While {positive_count} respondents expressed positive views, {negative_count} shared concerns or suggestions for improvement."
    elif positive_count > negative_count:
        summary += f" Respondents generally appreciated various aspects, with {positive_count} positive mentions."
    elif negative_count > positive_count:
        summary += f" Several areas for improvement were identified, with {negative_count} responses highlighting concerns."
    
    return {
        'summary': summary,
        'response_count': response_count,
        'avg_rating': None,
        'sentiment': sentiment,
        'keywords': keywords
    }

def main():
    st.title("📊 Feedback Form Summarizer")
    st.markdown("### Analyze and summarize feedback responses with AI-powered insights")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    if not api_key:
        st.warning("⚠️ No Gemini API key found. The app will work with basic analysis. For AI-powered insights, please set the GEMINI_API_KEY environment variable.")
    else:
        st.success("✅ AI analysis enabled with Gemini API")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload CSV files containing feedback responses"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        if not api_key:
            st.info("💡 To enable AI features, add your Gemini API key to the environment variables.")
    
    if uploaded_file is None:
        st.markdown("""
        ## 🚀 Welcome to the Feedback Form Summarizer!
        
        This application helps you analyze and summarize feedback form responses. Here's what it can do:
        
        ### ✨ Key Features:
        - **📈 Sentiment Analysis**: Detect positive, negative, and neutral responses
        - **🔍 Keyword Extraction**: Identify frequently mentioned topics and themes
        - **📝 Smart Summaries**: Generate concise summaries for each question
        - **📊 Rating Analysis**: Calculate averages for numeric rating questions
        - **🤖 AI-Powered**: Enhanced analysis with Gemini AI (when API key is provided)
        
        ### 📋 Supported File Format:
        - CSV files (.csv)
        
        ### 🏗️ Expected Data Structure:
        Your CSV file should contain columns with survey questions as headers and responses in rows.
        
        **Upload a CSV file using the sidebar to get started!**
        """)
        return
    
    # Process uploaded file
    try:
        # Read file content
        file_content = uploaded_file.getvalue().decode('utf-8')
        
        # Parse CSV
        headers, data = parse_csv_simple(file_content)
        
        if not headers or not data:
            st.error("❌ Could not parse the CSV file. Please check the format.")
            return
        
        st.success(f"📊 Successfully loaded {len(data)} responses with {len(headers)} questions")
        
        # Show data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.text("First few rows:")
            for i, row in enumerate(data[:5]):
                st.text(f"Row {i+1}: {row}")
            st.info(f"**Shape**: {len(data)} rows × {len(headers)} columns")
        
        # Process each question
        st.header("📊 Analysis Results")
        
        # Overall statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Responses", len(data))
        with col2:
            st.metric("Questions Analyzed", len(headers))
        with col3:
            st.metric("Analysis Method", "AI-Powered" if api_key else "Basic")
        
        # Analyze each question
        st.header("📝 Question-by-Question Analysis")
        
        # Add rate limiting notice
        if api_key:
            st.info("⏳ Processing questions with AI analysis. This may take a moment to respect API rate limits...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, question in enumerate(headers):
            if not question.strip():
                continue
            
            # Update progress
            progress = (i + 1) / len(headers)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing Q{i+1}: {question[:50]}...")
            
            # Get responses for this question (column)
            question_responses = [row[i] if i < len(row) else '' for row in data]
            
            # Analyze the question
            analysis = analyze_question_with_ai(question, question_responses, api_key)
            
            # Display results
            with st.expander(f"**Q{i+1}: {question}**", expanded=True):
                    
                    # Summary
                    st.subheader("📄 Summary")
                    st.write(analysis['summary'])
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Responses", analysis['response_count'])
                    with col2:
                        if analysis['avg_rating']:
                            st.metric("Average Rating", f"{analysis['avg_rating']:.2f}")
                        else:
                            st.metric("Response Type", "Text")
                    with col3:
                        response_rate = (analysis['response_count'] / len(data) * 100) if len(data) > 0 else 0
                        st.metric("Response Rate", f"{response_rate:.1f}%")
                    
                    # Sentiment analysis
                    if analysis['sentiment']:
                        st.subheader("😊 Sentiment Analysis")
                        sent_col1, sent_col2, sent_col3 = st.columns(3)
                        with sent_col1:
                            st.metric("Positive", analysis['sentiment'].get('Positive', 0))
                        with sent_col2:
                            st.metric("Neutral", analysis['sentiment'].get('Neutral', 0))
                        with sent_col3:
                            st.metric("Negative", analysis['sentiment'].get('Negative', 0))
                    
                    # Keywords
                    if analysis['keywords']:
                        st.subheader("🏷️ Key Themes")
                        st.write("**Top keywords mentioned:**")
                        keyword_text = " • ".join(analysis['keywords'])
                        st.write(keyword_text)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Generate overall insights
        if api_key:
            st.header("💡 Overall Insights")
            
            with st.spinner("🔍 Generating actionable insights..."):
                try:
                    import google.generativeai as genai
                    
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Collect all summaries for overall analysis
                    all_summaries = []
                    for i, question in enumerate(headers):
                        if question.strip():
                            question_responses = [row[i] if i < len(row) else '' for row in data]
                            analysis = analyze_question_with_ai(question, question_responses, api_key)
                            all_summaries.append(f"Q{i+1}: {question}\nSummary: {analysis['summary']}")
                    
                    combined_summary = "\n\n".join(all_summaries)
                    
                    prompt = f"""
                    Based on the following feedback analysis from multiple questions, provide key insights and actionable recommendations:

                    {combined_summary}

                    Please provide:
                    1. Top 3 key insights
                    2. Top 3 actionable recommendations
                    3. Overall sentiment assessment
                    4. Priority areas for improvement

                    Format as clear bullet points with markdown.
                    """
                    
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.warning(f"Could not generate overall insights: {str(e)}")
        
        # Download report
        st.header("⬇️ Download Results")
        
        # Generate text report
        report_lines = []
        report_lines.append("FEEDBACK FORM ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Responses: {len(data)}")
        report_lines.append(f"Total Questions: {len(headers)}")
        report_lines.append("")
        
        for i, question in enumerate(headers):
            if question.strip():
                question_responses = [row[i] if i < len(row) else '' for row in data]
                analysis = analyze_question_with_ai(question, question_responses, api_key)
                
                report_lines.append(f"Q{i+1}: {question}")
                report_lines.append("-" * 40)
                report_lines.append(f"Summary: {analysis['summary']}")
                report_lines.append(f"Response Count: {analysis['response_count']}")
                
                if analysis['sentiment']:
                    report_lines.append("Sentiment Distribution:")
                    for sentiment, count in analysis['sentiment'].items():
                        report_lines.append(f"  {sentiment}: {count}")
                
                if analysis['keywords']:
                    report_lines.append(f"Keywords: {', '.join(analysis['keywords'])}")
                
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="📄 Download Summary Report (TXT)",
            data=report_text,
            file_name=f"feedback_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check your CSV file format and try again.")

if __name__ == "__main__":
    main()
