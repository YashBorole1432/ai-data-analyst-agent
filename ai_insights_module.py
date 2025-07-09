# 🔮 AI-Powered Business Insights + Quick Share Feature

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
import plotly.express as px
import calendar
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib import colors
from datetime import datetime
import numpy as np # Added for optimize_dataframe
from typing import Optional
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from ai_insights_module import*

# File uploader
st.header("🔮 AI Business Insights Generator")
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

# Read and check the data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # ✅ Now df is your DataFrame

    if not df.empty:
        st.markdown("Generate meaningful business insights using GPT based on your uploaded dataset.")

        if st.button("🧠 Generate Insights"):
            with st.spinner("Thinking like an analyst..."):
                sample_data = df.sample(min(1000, len(df))).to_csv(index=False)

                prompt = f"""
You're a senior business analyst. Analyze the dataset below and generate actionable insights in bullet points:

- Highlight trends (e.g., top regions, products, time periods)
- Mention anomalies or outliers
- Give strategic recommendations for business
- Keep it short and relevant for business teams

DATA:
{sample_data}

INSIGHTS:
"""

                try:
                    openai.api_key = st.secrets["openai_api_key"]  # Use Streamlit secrets for security
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=700
                    )
                    insights = response.choices[0].message.content.strip()
                    st.success("✅ Insights Generated!")
                    st.markdown(insights)

                    # Downloadable insights text file
                    st.download_button(
                        label="📄 Download Insights as TXT",
                        data=insights,
                        file_name="business_insights.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ GPT Error: {e}")
    else:
        st.warning("⚠️ The uploaded CSV file is empty.")
else:
    st.info("📁 Please upload a CSV file first.")
