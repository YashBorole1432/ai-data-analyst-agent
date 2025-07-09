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
# Set the page configuration
st.set_page_config(page_title="AI Data Analyst", layout="wide")

# Title of the application
st.title("📊 AI Data Analyst")
# ================================
# 🧠 Memory Optimization Function
# ================================
@st.cache_data(show_spinner=False)
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes DataFrame memory usage by downcasting numeric types
    and converting low-cardinality object columns to 'category'.
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Downcast numeric types
    for col in df.select_dtypes(include=["int", "float"]).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if str(df[col].dtype).startswith("int"):
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype("uint8")
                elif col_max < 65535:
                    df[col] = df[col].astype("uint16")
                elif col_max < 4294967295:
                    df[col] = df[col].astype("uint32")
                elif col_max < 18446744073709551615: # uint64
                    df[col] = df[col].astype("uint64")
            else: # Signed integers
                if col_min > np.iinfo("int8").min and col_max < np.iinfo("int8").max:
                    df[col] = df[col].astype("int8")
                elif col_min > np.iinfo("int16").min and col_max < np.iinfo("int16").max:
                    df[col] = df[col].astype("int16")
                elif col_min > np.iinfo("int32").min and col_max < np.iinfo("int32").max:
                    df[col] = df[col].astype("int32")
                elif col_min > np.iinfo("int64").min and col_max < np.iinfo("int64").max:
                    df[col] = df[col].astype("int64")

        elif str(df[col].dtype).startswith("float"):
            df[col] = df[col].astype("float32") # float32 is usually sufficient

    # Convert object types with low unique count to category
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() / len(df) < 0.5: # If unique values are less than 50% of total rows
            df[col] = df[col].astype("category")

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    saved = initial_memory - optimized_memory
    st.success(f"✅ Optimized memory usage: {initial_memory:.2f} MB → {optimized_memory:.2f} MB ({saved:.2f} MB saved)")
    return df

# Function to clean and drop all-NA columns
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by dropping columns that are entirely NaN.
    """
    if df is None:
        raise ValueError("Received None instead of a DataFrame")
    df_cleaned = df.copy()
    initial_cols = df_cleaned.shape[1]
    df_cleaned.dropna(axis=1, how='all', inplace=True)
    if df_cleaned.shape[1] < initial_cols:
        st.info(f"🧹 Dropped {initial_cols - df_cleaned.shape[1]} columns that were entirely empty.")
    return df_cleaned

## Display memory usage (kept for potential direct use, though optimize_dataframe_memory prints it)
def memory_usage(df: pd.DataFrame) -> str:
    """Calculates and returns the memory usage of a DataFrame in MB."""
    mem = df.memory_usage(deep=True).sum() / (1024**2)  # in MB
    return f"{mem:.2f} MB"

@st.cache_data(show_spinner=False)
def load_csv_in_chunks(file_buffer, chunksize=5000, max_rows: Optional[int] = None):
    """
    Loads a CSV file in chunks, with an option to limit the total number of rows.
    Includes checks for empty files and invalid CSVs.
    """
    if file_buffer.size == 0:
        raise ValueError("Uploaded file is empty.")

    try:
        # Try reading just one line to see if it's valid
        # Use BytesIO to allow pandas to read from the buffer multiple times
        file_buffer.seek(0) # Reset buffer position
        first_row = pd.read_csv(file_buffer, nrows=1)
        if first_row.empty:
            raise ValueError("CSV file contains no data or no valid columns.")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or invalid.")
    except Exception as e:
        raise ValueError(f"Error reading first row of CSV: {e}")
    
    # Reset file position to start before reading chunks
    file_buffer.seek(0)

    chunks = []
    total_rows = 0
    progress_bar = st.progress(0)
    
    # Determine total file size for progress bar if max_rows is not set
    # This is a rough estimate and might not be perfectly accurate for progress
    total_file_size = file_buffer.seek(0, os.SEEK_END)
    file_buffer.seek(0) # Reset after getting size

    for i, chunk in enumerate(pd.read_csv(file_buffer, chunksize=chunksize)):
        chunks.append(chunk)
        total_rows += len(chunk)
        
        # Update progress bar
        # If max_rows is set, progress based on rows, otherwise based on file read position
        if max_rows:
            progress_val = min(total_rows / max_rows, 1.0)
        else:
            # Estimate progress based on file position (less reliable for CSVs)
            current_pos = file_buffer.tell()
            progress_val = min(current_pos / total_file_size, 1.0) if total_file_size > 0 else 0
        
        progress_bar.progress(progress_val)

        if max_rows and total_rows >= max_rows:
            break

    progress_bar.empty() # Clear the progress bar
    return pd.concat(chunks, ignore_index=True)

# Initialize df outside the conditional blocks to ensure it's always defined
df = None
uploaded_file = None # Ensure uploaded_file is also initialized

# File uploader for CSV files
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

    # Clear session state if no file is uploaded
    if 'df' in st.session_state:
        del st.session_state['df']
    if 'numeric_cols' in st.session_state:
        del st.session_state['numeric_cols']
    if 'categorical_cols' in st.session_state:
        del st.session_state['categorical_cols']

# Retrieve df and column lists from session state
df = st.session_state.get('df')
numeric_cols = st.session_state.get('numeric_cols', [])
categorical_cols = st.session_state.get('categorical_cols', [])



###AUto ML Model Integration###
st.header("🤖 Auto ML Model Integration")

if df is not None and not df.empty:
    st.markdown("### 🎯 Select the Target Column (What you want to predict)")
    target_col = st.selectbox("Target Column", df.columns)

    if target_col:
        # Drop rows with null in target
        df = df.dropna(subset=[target_col])

        # Validation: Check if target is numeric or has too many classes
        target_unique = df[target_col].nunique()
        if target_unique > 100:
            st.warning(f"⚠️ The selected column '{target_col}' has {target_unique} unique values. That’s too many for prediction.")
        else:
            # Prepare features
            features = df.drop(columns=[target_col])
            for col in features.select_dtypes(include='object').columns:
                features[col] = LabelEncoder().fit_transform(features[col].astype(str))

            X = features
            y = df[target_col]

            # Detect task type
            is_classification = y.nunique() <= 20 and y.dtype == 'object' or y.dtype.name == 'category'

            if is_classification:
                y = LabelEncoder().fit_transform(y)
                model = RandomForestClassifier()
                task_type = "Classification"
            else:
                if not np.issubdtype(y.dtype, np.number):
                    st.error("❌ Regression only works with numeric targets.")
                    st.stop()
                model = RandomForestRegressor()
                task_type = "Regression"

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Results
            st.subheader(f"📊 {task_type} Model Results")

            if is_classification:
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc*100:.2f}%")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                st.metric("R² Score", f"{r2:.2f}")
                st.metric("Mean Absolute Error", f"{mae:.2f}")

            # Predictions Preview
            st.subheader("🔍 Predictions Preview")
            preview_df = X_test.copy()
            preview_df['Actual'] = y_test
            preview_df['Predicted'] = y_pred
            st.dataframe(preview_df.head(20))

            # Download
            st.download_button("📥 Download Predictions CSV", preview_df.to_csv(index=False), "predictions.csv")

else:
    st.info("Please upload a valid dataset first.")


# 🔮 AI-Powered Business Insights + Quick Share Feature

import openai
import streamlit as st

st.header("🔮 AI Business Insights Generator")

if df is not None and not df.empty:
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
                openai.api_key = st.secrets["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"]  # Use Streamlit secrets for security
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
    st.info("📁 Please upload a CSV file first.")


# Only proceed with analysis if df is loaded
if df is not None:
    # Show paginated preview of the data
    st.subheader("🔍 Interactive Data Preview (Paged)")

    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    total_rows = len(df)
    total_pages = (total_rows - 1) // page_size + 1

    page = st.number_input("Select page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    st.caption(f"Showing rows {start_idx + 1} to {min(end_idx, total_rows)} of {total_rows}")
    st.dataframe(df.iloc[start_idx:end_idx])

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    # Null value count
    st.subheader("❗ Null Value Count")
    st.write(df.isnull().sum())

    # Section for data visualizations
    st.subheader("📊 Data Visualization")
    chart_type = st.selectbox("Choose a chart type", ["Histogram", "Bar Plot", "Box Plot"])

    # User controls for chart size
    width = st.slider("Figure Width", min_value=5, max_value=20, value=10)
    height = st.slider("Figure Height", min_value=3, max_value=15, value=5)

    # Generate selected chart
    if chart_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Select numeric column for histogram", numeric_cols, key="hist_col_select")
            fig, ax = plt.subplots(figsize=(width, height))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue')
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for histogram.")

    elif chart_type == "Bar Plot":
        if categorical_cols:
            col = st.selectbox("Select categorical column for bar plot", categorical_cols, key="bar_col_select")
            top_n = st.slider("Top N categories", min_value=5, max_value=50, value=20, key="bar_top_n")
            fig, ax = plt.subplots(figsize=(width, height))
            df[col].value_counts().head(top_n).plot(kind='bar', ax=ax, color='salmon')
            ax.set_title(f"Top {top_n} Most Frequent Values in '{col}'")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available for bar plot.")

    elif chart_type == "Box Plot":
        if numeric_cols:
            col = st.selectbox("Select numeric column for box plot", numeric_cols, key="box_col_select")
            fig, ax = plt.subplots(figsize=(width, height))
            sns.boxplot(y=df[col].dropna(), ax=ax, color='lightgreen')
            ax.set_title(f"Box Plot of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for box plot.")

    # ============================================
    # ✅ CLEANING & PREPROCESSING SECTION STARTS
    # ============================================
    st.subheader("🧹 Data Cleaning and Preprocessing")

    # 1. Remove duplicate rows
    if st.checkbox("Remove duplicate rows"):
        initial_shape = df.shape
        df.drop_duplicates(inplace=True)
        st.success(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
        st.session_state['df'] = df # Update session state after modification

    # 2. Handle missing values
    st.markdown("### 🔍 Missing Value Handling")
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if not missing.empty:
        st.write("Missing Values (%):")
        st.dataframe(missing)

        method = st.selectbox("Choose how to handle missing values", ["Do Nothing", "Drop Rows", "Fill with Mean/Mode"])

        if method == "Drop Rows":
            initial_rows = df.shape[0]
            df.dropna(inplace=True)
            st.success(f"Dropped {initial_rows - df.shape[0]} rows with missing values.")
            st.session_state['df'] = df # Update session state
            # Re-detect columns as some might become empty after dropping rows
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            st.session_state['numeric_cols'] = numeric_cols
            st.session_state['categorical_cols'] = categorical_cols

        elif method == "Fill with Mean/Mode":
            for col in df.columns: # Iterate through all columns
                if df[col].isnull().sum() > 0: # Check if column has missing values
                    if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
            st.success("Filled missing values with mean/mode.")
            st.session_state['df'] = df # Update session state
    else:
        st.info("No missing values detected.")

    # 3. Convert data types
    st.markdown("### 🔧 Convert Data Types")
    # Ensure we use the current columns from the potentially modified df
    current_cols = df.columns.tolist()
    convert_cols = st.multiselect("Select columns to convert to numeric", current_cols)
    if convert_cols:
        for col in convert_cols:
            # Use errors='coerce' to turn unconvertible values into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Optionally, fill NaNs created by coercion if desired
            if df[col].isnull().any():
                st.warning(f"Column '{col}' had non-numeric values converted to NaN. Consider handling these NaNs.")
        st.success(f"Converted selected columns to numeric type.")
        st.session_state['df'] = df # Update session state
        # Re-detect columns after type conversion
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        st.session_state['numeric_cols'] = numeric_cols
        st.session_state['categorical_cols'] = categorical_cols

    # 4. Rename columns (Optional)
    st.markdown("### ✏️ Rename Columns")
    if st.checkbox("Rename columns"):
        col_to_rename = st.selectbox("Select a column to rename", df.columns.tolist(), key="rename_select_col")
        new_name = st.text_input("Enter new name for the column", key="rename_new_name")
        if st.button("Rename Column", key="rename_button"):
            if col_to_rename and new_name and col_to_rename in df.columns:
                df.rename(columns={col_to_rename: new_name}, inplace=True)
                st.success(f"Renamed '{col_to_rename}' to '{new_name}'")
                st.session_state['df'] = df # Update session state
                # Re-detect columns after renaming
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                categorical_cols = df.select_dtypes(include='object').columns.tolist()
                st.session_state['numeric_cols'] = numeric_cols
                st.session_state['categorical_cols'] = categorical_cols
            else:
                st.warning("Please select a column and enter a new name.")

    # 5. Show cleaned data preview
    st.subheader("📄 Cleaned Data Preview")
    st.dataframe(df.head())

    # 6. Optional: Download cleaned data
    st.download_button("📥 Download Cleaned Data as CSV", df.to_csv(index=False).encode('utf-8'), file_name="cleaned_data.csv", mime="text/csv")

    # =========================================
    # 🔮 One-Click Insight Generator
    # =========================================
    st.markdown("---")
    st.header("🔮 One-Click Insight Generator")

    # Ensure OpenAI API key is set (consider using st.secrets for production)
    # openai.api_key = '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'
    # It's better to get the API key from Streamlit secrets or environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    if not openai_api_key:
        st.warning("OpenAI API key not found. Please set it in your environment variables or Streamlit secrets.")
    else:
        openai.api_key = openai_api_key

        st.markdown("Generate smart summaries, KPIs, trends, and suggestions for your dataset using AI.")

        if st.button("🧠 Generate Insights"):
            if df.empty:
                st.warning("The DataFrame is empty. Cannot generate insights.")
            elif not openai_api_key:
                st.error("OpenAI API key is not configured. Cannot generate insights.")
            else:
                with st.spinner("Generating AI-powered insights..."):
                    try:
                        # Sample a portion of the dataset for performance
                        # Ensure the sample size doesn't exceed the DataFrame length
                        sample_size = min(len(df), 1000)
                        data_sample = df.sample(sample_size, random_state=42).to_csv(index=False) # Added random_state for reproducibility

                        # Prompt to GPT
                        prompt = f"""
You are an expert data analyst. Based on the following dataset (sample), generate a detailed set of insights that include:

1. Key patterns, trends, or anomalies.
2. Summary statistics or metrics worth highlighting.
3. Any outliers or performance spikes.
4. Potential business or operational recommendations.
5. Suggestions for further analysis or next steps.

DATA SAMPLE:
{data_sample}

INSIGHTS:
"""
                        # Call OpenAI
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=700
                        )

                        answer = response.choices[0].message.content.strip()
                        st.success("✅ Insights Generated!")
                        st.markdown(f"#### 📋 AI Summary")
                        st.markdown(answer)

                    except openai.error.AuthenticationError:
                        st.error("❌ OpenAI API key is invalid or not configured correctly. Please check your API key.")
                    except openai.error.RateLimitError:
                        st.error("❌ You've hit your OpenAI API rate limit. Please wait or upgrade your plan.")
                    except Exception as e:
                        st.error(f"❌ Error generating insights: {e}")

    #_____________________
    ###KPI Card Builder##
    #_____________________
    st.subheader("📌 Custom KPI Card Builder")

    if not df.empty and numeric_cols:
        col = st.selectbox("Select a metric column", numeric_cols, key="kpi_col")
        agg_func = st.selectbox("Select aggregation", ["Sum", "Mean", "Max", "Min", "Count"], key="agg_func")

        value = None
        # Calculate KPI based on aggregation selected
        if agg_func == "Sum":
            value = df[col].sum()
        elif agg_func == "Mean":
            value = df[col].mean()
        elif agg_func == "Max":
            value = df[col].max()
        elif agg_func == "Min":
            value = df[col].min()
        elif agg_func == "Count":
            value = df[col].count()

        if value is not None:
            # Format the number for display
            formatted_value = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
            st.metric(label=f"{agg_func} of {col}", value=formatted_value)
        else:
            st.warning("Could not calculate KPI. Check column data or aggregation.")
    else:
        st.info("No numeric columns available for KPI card or DataFrame is empty.")


    # =========================================
    # 📊 Interactive Dashboard
    # =========================================
    st.markdown("---")
    st.header("📊 Interactive Dashboard")

    # Sidebar filters for numeric and categorical columns
    st.sidebar.header("🔎 Filter Your Data")

    # Initialize filtered_df with the full df at the start of this section
    filtered_df = df.copy()
    
    # Filters dictionary
    filter_conditions = []

    # Numeric filters with slider range
    if numeric_cols:
        for col in numeric_cols:
            # Ensure min/max values are valid numbers before creating slider
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                # Handle cases where min_val == max_val to prevent slider error
                if min_val == max_val:
                    st.sidebar.write(f"**{col}**: All values are {min_val:.2f}")
                    filter_conditions.append(df[col] == min_val)
                else:
                    selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val), key=f"num_filter_{col}")
                    filter_conditions.append((df[col] >= selected_range[0]) & (df[col] <= selected_range[1]))
            else:
                st.sidebar.info(f"Skipping numeric filter for '{col}' as it's not purely numeric.")

    # Categorical filters with multiselect
    if categorical_cols:
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique().tolist()
            if unique_vals: # Only show multiselect if there are unique values
                selected_vals = st.sidebar.multiselect(f"Select {col}", options=unique_vals, default=unique_vals, key=f"cat_filter_{col}")
                filter_conditions.append(df[col].isin(selected_vals))
            else:
                st.sidebar.info(f"No unique values in categorical column '{col}' to filter.")

    # Apply all filters together
    if filter_conditions:
        # Combine all conditions using & (AND)
        combined_condition = pd.Series([True] * len(df), index=df.index)
        for condition in filter_conditions:
            combined_condition = combined_condition & condition
        filtered_df = df.loc[combined_condition]
    else:
        filtered_df = df.copy() # If no filters, use the original df

    st.write(f"### Filtered Data: {filtered_df.shape[0]} rows")

    # Show filtered data preview
    st.dataframe(filtered_df)

    # Summary metrics (cards style)
    st.markdown("### Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", filtered_df.shape[0])
    with col2:
        st.metric("Total Columns", filtered_df.shape[1])
    with col3:
        if numeric_cols and not filtered_df.empty:
            # Ensure the column exists in filtered_df and is numeric
            if numeric_cols[0] in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[numeric_cols[0]]):
                st.metric(f"Mean of {numeric_cols[0]}", f"{filtered_df[numeric_cols[0]].mean():.2f}")
            else:
                st.write("Mean: N/A (Column not found or not numeric in filtered data)")
        else:
            st.write("Mean: N/A (No numeric columns or empty filtered data)")
    with col4:
        if numeric_cols and not filtered_df.empty:
            if numeric_cols[0] in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[numeric_cols[0]]):
                st.metric(f"Median of {numeric_cols[0]}", f"{filtered_df[numeric_cols[0]].median():.2f}")
            else:
                st.write("Median: N/A (Column not found or not numeric in filtered data)")
        else:
            st.write("Median: N/A (No numeric columns or empty filtered data)")

    # Visualization panel
    st.markdown("### Visualizations")

    # Stacked Bar Chart
    if len(categorical_cols) >= 2 and not filtered_df.empty:
        st.subheader("Stacked Bar Chart")
        cat_col1 = st.selectbox("Select primary categorical column (x-axis)", categorical_cols, key="stack_cat1")
        cat_col2 = st.selectbox("Select secondary categorical column (stack)", [c for c in categorical_cols if c != cat_col1], key="stack_cat2")

        if cat_col1 and cat_col2:
            # Prepare data for stacked bar chart, handling potential NaNs
            stacked_data = filtered_df.groupby([cat_col1, cat_col2]).size().unstack(fill_value=0)
            if not stacked_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                stacked_data.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
                ax.set_ylabel("Count")
                ax.set_title(f"Stacked Bar Chart of {cat_col1} by {cat_col2}")
                ax.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout() # Adjust layout to prevent labels from overlapping
                st.pyplot(fig)
            else:
                st.warning(f"No data to display for stacked bar chart with selected columns: {cat_col1}, {cat_col2}")
        else:
            st.info("Please select two different categorical columns for the stacked bar chart.")
    else:
        st.info("Need at least two categorical columns and non-empty filtered data for stacked bar chart.")

    # Violin Plot for distribution analysis
    if numeric_cols and categorical_cols and not filtered_df.empty:
        st.subheader("🎻 Violin Plot")
        num_col = st.selectbox("Select numeric column for violin plot", numeric_cols, key="violin_num_col")
        cat_col = st.selectbox("Select categorical column for violin plot", categorical_cols, key="violin_cat_col")

        if cat_col in filtered_df.columns and num_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[num_col]):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(x=filtered_df[cat_col], y=filtered_df[num_col], ax=ax, palette="Set2")
            ax.set_title(f"Violin Plot of {num_col} by {cat_col}")
            plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"⚠️ Invalid selection: `{cat_col}` or `{num_col}` not suitable for violin plot in filtered data.")
    else:
        st.info("Need numeric and categorical columns and non-empty filtered data for violin plot.")

    # Bar Plot for category distribution
    if categorical_cols and not filtered_df.empty:
        st.subheader("📊 Bar Plot of Categorical Column")
        bar_col = st.selectbox("Select categorical column for bar plot", categorical_cols, key="bar_plot_select_2")
        top_n = st.slider("Top N categories", min_value=5, max_value=50, value=20, key="bar_plot_top_n_2")

        bar_data = filtered_df[bar_col].value_counts().head(top_n)
        if not bar_data.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=bar_data.values, y=bar_data.index, palette="magma", ax=ax)
            ax.set_title(f"Top {top_n} Values in '{bar_col}'")
            ax.set_xlabel("Count")
            ax.set_ylabel(bar_col)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"No data to display for bar plot with selected column: {bar_col}")
    else:
        st.info("Need categorical columns and non-empty filtered data for bar plot.")

    # Histogram
    if numeric_cols and not filtered_df.empty:
        st.subheader("📉 Histogram of Numeric Column")
        hist_col = st.selectbox("Select numeric column for histogram", numeric_cols, key="hist_column_2")
        if hist_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[hist_col]):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(filtered_df[hist_col].dropna(), kde=True, color="skyblue", ax=ax)
            ax.set_title(f"Histogram of {hist_col}")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"⚠️ Invalid selection: `{hist_col}` not suitable for histogram in filtered data.")
    else:
        st.info("Need numeric columns and non-empty filtered data for histogram.")

    # Dashboard with Plotly (E-commerce specific)
    st.markdown("---")
    st.header("📊 E-commerce Dashboard")

    # Show metrics if columns exist
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'Quantity' in df.columns and pd.api.types.is_numeric_dtype(df['Quantity']):
            st.metric("Sum of Quantity", int(df['Quantity'].sum()))
        else:
            st.warning("❗ 'Quantity' column not found or not numeric.")

    with col2:
        if 'Profit' in df.columns and pd.api.types.is_numeric_dtype(df['Profit']):
            st.metric("Sum of Profit", f"{int(df['Profit'].sum()):,} ₹")
        else:
            st.warning("❗ 'Profit' column not found or not numeric.")

    with col3:
        if 'Amount' in df.columns and pd.api.types.is_numeric_dtype(df['Amount']):
            st.metric("Sum of Amount", f"{int(df['Amount'].sum()):,} ₹")
        else:
            st.warning("❗ 'Amount' column not found or not numeric.")

    with col4:
        if 'PaymentMode' in df.columns and not df['PaymentMode'].empty:
            mode = df['PaymentMode'].mode()[0]
            st.metric("Top Payment Mode", mode)
        else:
            st.warning("❗ 'PaymentMode' column not found or is empty.")

    # Pie Charts
    st.markdown("### 📈 Quantity Distribution")

    col1, col2 = st.columns(2)

    with col1:
        if {'Category', 'Quantity'}.issubset(df.columns) and pd.api.types.is_numeric_dtype(df['Quantity']):
            qty_by_cat = df.groupby("Category")['Quantity'].sum().reset_index()
            if not qty_by_cat.empty:
                fig1 = px.pie(qty_by_cat, names='Category', values='Quantity', hole=0.4, title="Quantity by Category")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("No data for Quantity by Category pie chart.")
        else:
            st.warning("❗ 'Category' or 'Quantity' column not found or 'Quantity' is not numeric.")

    with col2:
        if {'PaymentMode', 'Quantity'}.issubset(df.columns) and pd.api.types.is_numeric_dtype(df['Quantity']):
            qty_by_pay = df.groupby("PaymentMode")['Quantity'].sum().reset_index()
            if not qty_by_pay.empty:
                fig2 = px.pie(qty_by_pay, names='PaymentMode', values='Quantity', hole=0.4, title="Quantity by Payment Mode")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No data for Quantity by Payment Mode pie chart.")
        else:
            st.warning("❗ 'PaymentMode' or 'Quantity' column not found or 'Quantity' is not numeric.")

    # Profit by Sub-Category
    if {'Sub-Category', 'Profit'}.issubset(df.columns) and pd.api.types.is_numeric_dtype(df['Profit']):
        st.markdown("### 💹 Profit by Sub-Category")
        profit_by_sub = df.groupby("Sub-Category")['Profit'].sum().reset_index().sort_values(by='Profit')
        if not profit_by_sub.empty:
            fig3 = px.bar(profit_by_sub, x='Profit', y='Sub-Category', orientation='h', title="Profit by Sub-Category", color='Profit')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No data for Profit by Sub-Category bar chart.")
    else:
        st.warning("❗ 'Sub-Category' or 'Profit' column not found or 'Profit' is not numeric.")

    # Monthly Profit
    if {'Date', 'Profit'}.issubset(df.columns) and pd.api.types.is_numeric_dtype(df['Profit']):
        st.markdown("### 📅 Profit by Month")
        # Convert 'Date' column to datetime, coercing errors
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows where 'Date' became NaT (Not a Time) due to coercion
        df_valid_dates = df.dropna(subset=['Date'])

        if not df_valid_dates.empty:
            df_valid_dates['Month'] = df_valid_dates['Date'].dt.strftime('%B')
            monthly_profit = df_valid_dates.groupby('Month')['Profit'].sum().reset_index()
            
            # Ensure months are ordered correctly
            monthly_profit['Month'] = pd.Categorical(monthly_profit['Month'], categories=list(calendar.month_name)[1:], ordered=True)
            monthly_profit = monthly_profit.sort_values('Month')

            if not monthly_profit.empty:
                fig4 = px.bar(monthly_profit, x='Month', y='Profit', title="Profit by Month", color='Profit')
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("No data for Monthly Profit bar chart after date processing.")
        else:
            st.warning("❗ No valid dates found in 'Date' column after conversion.")
    else:
        st.warning("❗ 'Date' or 'Profit' column not found or 'Profit' is not numeric.")

    ##Report Generator (Excel)
    st.header("📝 Generate Downloadable Report (Excel)")
    if st.button("Generate Excel Report"):
        if not df.empty:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            buffer.seek(0) # Rewind the buffer
            st.download_button(
                label="📥 Download Excel Report",
                data=buffer.getvalue(),
                file_name="data_report.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            st.warning("DataFrame is empty, cannot generate Excel report.")

    st.markdown("## 🔥 Correlation Heatmap")
    if numeric_cols and len(numeric_cols) > 1 and not df.empty:
        st.subheader("Correlation Heatmap of Numeric Columns")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Need at least two numeric columns and non-empty data to display a correlation heatmap.")

    # PDF Report Generator
    def generate_industry_pdf(df_to_report: pd.DataFrame):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        heading = ParagraphStyle(name='Heading', parent=styles['Heading2'], fontSize=13, spaceAfter=12)
        normal = ParagraphStyle(name='Normal', parent=styles['Normal'], fontSize=10)
        
        elements = []

        # -----------------------
        # 📘 Cover Page
        # -----------------------
        elements.append(Paragraph("📊 AI Data Analyst Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"📅 Date: {datetime.now().strftime('%B %d, %Y')}", normal))
        elements.append(Paragraph(f"🧾 Total Records: {df_to_report.shape[0]} | Columns: {df_to_report.shape[1]}", normal))
        elements.append(Paragraph("🔍 This report provides automated analysis on uploaded CSV data using industry best practices.", normal))
        elements.append(PageBreak())

        # -----------------------
        # 🧭 Table of Contents
        # -----------------------
        elements.append(Paragraph("📚 Table of Contents", styles['Heading1']))
        toc = [
            "1. Dataset Summary",
            "2. Missing Value Analysis",
            "3. Categorical Columns Overview",
            "4. Numeric Summary Stats",
            "5. Key Observations"
        ]
        for item in toc:
            elements.append(Paragraph(item, normal))
        elements.append(PageBreak())

        # -----------------------
        # 1. Dataset Summary
        # -----------------------
        elements.append(Paragraph("1️⃣ Dataset Summary", heading))
        summary = {
            "Total Rows": len(df_to_report),
            "Total Columns": len(df_to_report.columns),
            "Missing Values (Total)": df_to_report.isnull().sum().sum(),
            "Numeric Columns": len(df_to_report.select_dtypes(include='number').columns),
            "Categorical Columns": len(df_to_report.select_dtypes(include='object').columns),
        }
        summary_table = [["Metric", "Value"]] + [[k, str(v)] for k, v in summary.items()]
        table = Table(summary_table)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.black)
        ]))
        elements.append(table)
        elements.append(PageBreak())

        # -----------------------
        # 2. Missing Value Analysis
        # -----------------------
        elements.append(Paragraph("2️⃣ Missing Value Analysis", heading))
        missing = df_to_report.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            mv_table = [["Column", "Missing Count", "Missing %"]]
            for col in missing.index:
                percent = round(100 * df_to_report[col].isnull().mean(), 2)
                mv_table.append([col, missing[col], f"{percent}%"])
            table = Table(mv_table)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.black),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("✅ No missing values found in this dataset.", normal))
        elements.append(PageBreak())

        # -----------------------
        # 3. Categorical Columns
        # -----------------------
        elements.append(Paragraph("3️⃣ Top Categories in Categorical Columns", heading))
        cat_cols_report = df_to_report.select_dtypes(include='object').columns.tolist()
        if cat_cols_report:
            for col in cat_cols_report:
                top_vals = df_to_report[col].value_counts().head(5)
                if not top_vals.empty:
                    elements.append(Paragraph(f"🔸 {col}", styles['h4'])) # Use h4 style
                    cat_table = [["Value", "Count"]] + [[idx, val] for idx, val in top_vals.items()]
                    ct = Table(cat_table)
                    ct.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                    ]))
                    elements.append(ct)
                    elements.append(Spacer(1, 6))
                else:
                    elements.append(Paragraph(f"ℹ️ Column '{col}' has no unique categorical values.", normal))
        else:
            elements.append(Paragraph("ℹ️ No categorical columns found.", normal))
        elements.append(PageBreak())

        # -----------------------
        # 4. Numeric Summary
        # -----------------------
        elements.append(Paragraph("4️⃣ Numeric Columns Summary (Descriptive Stats)", heading))
        num_df_report = df_to_report.select_dtypes(include='number')
        if not num_df_report.empty:
            stats_df = num_df_report.describe().round(2).reset_index()
            stats_table = [stats_df.columns.tolist()] + stats_df.values.tolist()
            table = Table(stats_table, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.black),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("ℹ️ No numeric columns found.", normal))
        elements.append(PageBreak())

        # -----------------------
        # 5. Key Observations (Optional)
        # -----------------------
        elements.append(Paragraph("5️⃣ Key Observations", heading))
        obs = [
            f"- Dataset contains {df_to_report.shape[0]} rows and {df_to_report.shape[1]} columns.",
            f"- {len(missing)} columns have missing values." if not missing.empty else "- No columns have missing values.",
            f"- {len(cat_cols_report)} categorical columns detected.",
            f"- {len(num_df_report.columns)} numeric columns detected.",
        ]
        if cat_cols_report and not df_to_report[cat_cols_report[0]].empty:
            obs.append(f"- Most frequent value in '{cat_cols_report[0]}' is '{df_to_report[cat_cols_report[0]].mode()[0]}'.")
        else:
            obs.append("- No categorical columns or empty categorical data to determine most frequent value.")

        for item in obs:
            elements.append(Paragraph(f"• {item}", normal))

        # Export
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    st.header("📝 Generate Downloadable Report (PDF)")
    if st.button("📄 Download Industry Report"):
        if df is not None and not df.empty:
            with st.spinner("Generating PDF report..."):
                pdf_buffer = generate_industry_pdf(df)
                st.download_button(
                    label="📥 Download Industry PDF Report",
                    data=pdf_buffer,
                    file_name="AI_Industry_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Please upload and load a dataset first to generate the PDF report.")

else:
    st.info("Please upload a CSV or Excel file to begin analysis.")

