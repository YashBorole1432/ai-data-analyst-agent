import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os

# openai.api_key = st.secrets["openai"]["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"]

# openai.api_key = st.secrets["openai"]["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"]


# Set the page configuration
st.set_page_config(page_title="AI Data Analyst", layout="wide")

# Title of the application
st.title("📊 AI Data Analyst")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Show preview of the data
    st.subheader("🔍 Preview of Data")
    st.dataframe(df.head())

    # Detect numeric and categorical columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

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
            col = st.selectbox("Select numeric column for histogram", numeric_cols)
            fig, ax = plt.subplots(figsize=(width, height))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue')
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for histogram.")

    elif chart_type == "Bar Plot":
        if categorical_cols:
            col = st.selectbox("Select categorical column for bar plot", categorical_cols)
            top_n = st.slider("Top N categories", min_value=5, max_value=50, value=20)
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
            col = st.selectbox("Select numeric column for box plot", numeric_cols)
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

    # 2. Handle missing values
    st.markdown("### 🔍 Missing Value Handling")
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if not missing.empty:
        st.write("Missing Values (%):")
        st.dataframe(missing)

        method = st.selectbox("Choose how to handle missing values", ["Do Nothing", "Drop Rows", "Fill with Mean/Mode"])

        if method == "Drop Rows":
            df.dropna(inplace=True)
            st.success("Dropped rows with missing values.")

        elif method == "Fill with Mean/Mode":
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype == 'object':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        df[col].fillna(df[col].mean(), inplace=True)
            st.success("Filled missing values with mean/mode.")
    else:
        st.info("No missing values detected.")

    # 3. Convert data types
    st.markdown("### 🔧 Convert Data Types")
    convert_cols = st.multiselect("Select columns to convert to numeric", df.columns.tolist())
    if convert_cols:
        for col in convert_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        st.success(f"Converted selected columns to numeric type.")

    # 4. Rename columns (Optional)
    st.markdown("### ✏️ Rename Columns")
    if st.checkbox("Rename columns"):
        col_to_rename = st.selectbox("Select a column to rename", df.columns.tolist())
        new_name = st.text_input("Enter new name for the column")
        if st.button("Rename"):
            df.rename(columns={col_to_rename: new_name}, inplace=True)
            st.success(f"Renamed '{col_to_rename}' to '{new_name}'")

    # 5. Show cleaned data preview
    st.subheader("📄 Cleaned Data Preview")
    st.dataframe(df.head())

    # 6. Optional: Download cleaned data
    st.download_button("📥 Download Cleaned Data as CSV", df.to_csv(index=False), file_name="cleaned_data.csv")

else:
    st.info("📁 Please upload a CSV file to begin analysis.")


###****Implement The Open AI to Chat with Data 
# =========================================
# 🤖 GPT-Powered Natural Language Q&A
# =========================================
st.subheader("🤖 Ask AI About Your Data")

# Set your OpenAI API key securely
openai.api_key = st.secrets["openai"]["api_key"]

# User input box for a question
question = st.text_input("💬 Ask a question about your dataset (e.g., 'Which category has the highest sales?')")

if question and not df.empty:
    with st.spinner("Analyzing..."):
        try:
            # Prepare data sample
            data_sample = df.head(20).to_csv(index=False)

            # Prompt with formatted question and data
            prompt = f"""
You are a helpful data analyst. Based on the following data sample, answer the user's question clearly and concisely.

DATA SAMPLE:
{data_sample}

QUESTION:
{question}

ANSWER:
"""

            # Call OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )

            # Show result
            answer = response.choices[0].message.content.strip()
            st.success(answer)

        except Exception as e:
            st.error(f"❌ Error: {e}")

###***Adding The Power-Bi Dashbaord's 
if uploaded_file is not None:

    st.markdown("---")
    st.header("📊 Interactive Dashboard")

    # Sidebar filters for numeric and categorical columns
    st.sidebar.header("🔎 Filter Your Data")

    # Filters dictionary
    filter_conditions = []

    # Numeric filters with slider range
    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
        filter_conditions.append((df[col] >= selected_range[0]) & (df[col] <= selected_range[1]))

    # Categorical filters with multiselect
    for col in categorical_cols:
        unique_vals = df[col].dropna().unique().tolist()
        selected_vals = st.sidebar.multiselect(f"Select {col}", options=unique_vals, default=unique_vals)
        filter_conditions.append(df[col].isin(selected_vals))

    # Apply all filters together
    if filter_conditions:
        filtered_df = df.loc[pd.concat(filter_conditions, axis=1).all(axis=1)]
    else:
        filtered_df = df.copy()

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
        if numeric_cols:
            st.metric("Mean of first numeric col", f"{filtered_df[numeric_cols[0]].mean():.2f}")
        else:
            st.write("Mean: N/A")
    with col4:
        if numeric_cols:
            st.metric("Median of first numeric col", f"{filtered_df[numeric_cols[0]].median():.2f}")
        else:
            st.write("Median: N/A")

    # Visualization panel
    st.markdown("### Visualizations")

    # Select two numeric columns to plot scatter plot
    if len(categorical_cols) >= 2:
     st.subheader("Stacked Bar Chart")

    # Select two categorical columns for stacking
    cat_col1 = st.selectbox("Select primary categorical column (x-axis)", categorical_cols, key="stack_cat1")
    cat_col2 = st.selectbox("Select secondary categorical column (stack)", categorical_cols, key="stack_cat2")

    # Prepare data for stacked bar chart
    stacked_data = filtered_df.groupby([cat_col1, cat_col2]).size().unstack(fill_value=0)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    stacked_data.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_ylabel("Count")
    ax.set_title(f"Stacked Bar Chart of {cat_col1} by {cat_col2}")
    ax.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
else:
    st.info("Need at least two categorical columns for stacked bar chart.")



    # Show bar chart for a selected categorical column
   # Violin Plot for distribution analysis
if numeric_cols and categorical_cols:
    st.subheader("🎻 Violin Plot")
    num_col = st.selectbox("Select numeric column for violin plot", numeric_cols)
    cat_col = st.selectbox("Select categorical column for violin plot", categorical_cols)

    if cat_col in filtered_df.columns and num_col in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(x=filtered_df[cat_col], y=filtered_df[num_col], ax=ax, palette="Set2")
        ax.set_title(f"Violin Plot of {num_col} by {cat_col}")
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ Invalid selection: `{cat_col}` or `{num_col}` not in dataset.")

# Bar Plot for category distribution (moved outside the violin logic)
if categorical_cols:
    st.subheader("📊 Bar Plot of Categorical Column")
    bar_col = st.selectbox("Select categorical column for bar plot", categorical_cols)
    top_n = st.slider("Top N categories", min_value=5, max_value=50, value=20)

    bar_data = filtered_df[bar_col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=bar_data.values, y=bar_data.index, palette="magma", ax=ax)
    ax.set_title(f"Top {top_n} Values in '{bar_col}'")
    ax.set_xlabel("Count")
    ax.set_ylabel(bar_col)
    st.pyplot(fig)

# Histogram
if numeric_cols:
    st.subheader("📉 Histogram of Numeric Column")
    hist_col = st.selectbox("Select numeric column for histogram", numeric_cols, key="hist_column")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df[hist_col].dropna(), kde=True, color="skyblue", ax=ax)
    ax.set_title(f"Histogram of {hist_col}")
    st.pyplot(fig)



##*#** TRY TO ADD DASHBOARD DIRECTLTTTT
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

uploaded_file = st.file_uploader("Upload your E-commerce CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Sum of Quantity", int(df['Quantity'].sum()))

with col2:
    st.metric("Sum of Profit", f"{int(df['Profit'].sum()):,} ₹")

with col3:
    st.metric("Sum of Amount", f"{int(df['Amount'].sum()):,} ₹")

with col4:
    mode = df['PaymentMode'].mode()[0]
    st.metric("Top Payment Mode", mode)
# Sum of Quantity by Category
qty_by_cat = df.groupby("Category")['Quantity'].sum().reset_index()
fig1 = px.pie(qty_by_cat, names='Category', values='Quantity', hole=0.4, title="Quantity by Category")

# Quantity by PaymentMode
qty_by_pay = df.groupby("PaymentMode")['Quantity'].sum().reset_index()
fig2 = px.pie(qty_by_pay, names='PaymentMode', values='Quantity', hole=0.4, title="Quantity by Payment Mode")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
profit_by_sub = df.groupby("Sub-Category")['Profit'].sum().reset_index().sort_values(by='Profit')
fig3 = px.bar(profit_by_sub, x='Profit', y='Sub-Category', orientation='h', title="Profit by Sub-Category", color='Profit')
st.plotly_chart(fig3, use_container_width=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.strftime('%B')
monthly_profit = df.groupby('Month')['Profit'].sum().reset_index()

# Optional: Sort months correctly
import calendar
monthly_profit['Month'] = pd.Categorical(monthly_profit['Month'], categories=calendar.month_name[1:], ordered=True)
monthly_profit = monthly_profit.sort_values('Month')

fig4 = px.bar(monthly_profit, x='Month', y='Profit', title="Profit by Month", color='Profit')
st.plotly_chart(fig4, use_container_width=True)
