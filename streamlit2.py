# ========================================================================
# STREAMLIT INTRODUCTION SESSION
# Covers all major components in one interactive app
# ========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================================
# 1️⃣ PAGE CONFIGURATION
# ========================================================================
st.set_page_config(
    page_title="Streamlit Training Session",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Introduction
st.title("🎓 Streamlit — Interactive Python Web Apps")
st.header("Session: Introduction to Streamlit")
st.markdown("""
Streamlit is a Python library that turns your scripts into **interactive web apps** in minutes!  
In this demo, we’ll explore all key Streamlit features step by step:
1. **Text and display elements**
2. **Input widgets**
3. **Data display**
4. **Charts and visualizations**
5. **Layout and interactivity**
6. **File operations**
""")
st.divider()

# ========================================================================
# 2️⃣ TEXT & DISPLAY ELEMENTS
# ========================================================================
st.header("1️⃣ Text and Display Elements")

st.write("Streamlit supports text, markdown, LaTeX, and code blocks.")
st.markdown("**This is bold text**, *this is italic*, and `this is inline code`.")
st.latex(r"E = mc^2")
st.code("print('Hello Streamlit!')", language='python')

st.info("ℹ️ Use st.write() when unsure — it can display almost anything!")

st.divider()

# ========================================================================
# 3️⃣ INPUT WIDGETS
# ========================================================================
st.header("2️⃣ Input Widgets")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Text Inputs")
    name = st.text_input("Enter your name", "Zainab")
    age = st.number_input("Enter your age", min_value=0, max_value=100, value=22)
    gender = st.radio("Select your gender", ["Female", "Male", "Other"], horizontal=True)
    
with col2:
    st.subheader("📅 Date & Selections")
    date = st.date_input("Select a date")
    hobby = st.multiselect("Choose your hobbies", ["Reading", "Coding", "Music", "Sports"])
    rating = st.slider("Rate your interest in Data Science", 0, 10, 7)

st.success(f"👋 Welcome **{name}**! You rated Data Science {rating}/10.")
st.divider()

# ========================================================================
# 4️⃣ DATA DISPLAY
# ========================================================================
st.header("3️⃣ Displaying Data")

st.write("Let’s create a sample dataset and display it using different methods:")

# Create DataFrame
data = pd.DataFrame({
    "Name": ["Ali", "Sara", "Bilal", "Fatima", "Ahmed"],
    "Age": [24, 30, 27, 22, 28],
    "Score": [88, 75, 92, 85, 79]
})

st.subheader("📋 DataFrame Display")
st.dataframe(data, use_container_width=True)

st.subheader("📊 Metrics Example")
col1, col2, col3 = st.columns(3)
col1.metric("Average Age", f"{data['Age'].mean():.1f}")
col2.metric("Highest Score", f"{data['Score'].max()}")
col3.metric("Total Students", len(data))

st.divider()

# ========================================================================
# 5️⃣ CHARTS & VISUALIZATIONS
# ========================================================================
st.header("4️⃣ Visualizations")

st.write("Streamlit supports built-in charts and custom Matplotlib or Seaborn plots.")

# Built-in line chart
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Math", "Science", "English"])
st.line_chart(chart_data)

# Matplotlib example
fig, ax = plt.subplots()
ax.hist(data["Score"], bins=5, color="skyblue", edgecolor="black")
ax.set_title("Distribution of Scores")
ax.set_xlabel("Score")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Seaborn example
st.subheader("🎨 Seaborn Heatmap Example")
corr = chart_data.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.divider()

# ========================================================================
# 6️⃣ LAYOUT & INTERACTIVITY
# ========================================================================
st.header("5️⃣ Layout & Interactivity")

with st.sidebar:
    st.header("⚙️ Sidebar Filters")
    department = st.selectbox("Choose Department", ["IT", "HR", "Finance"])
    show_data = st.checkbox("Show sample data", value=True)

if show_data:
    st.write(f"Showing data for **{department} Department**")
    st.dataframe(data.sample(3))

# Tabs
tab1, tab2 = st.tabs(["📈 Statistics", "💡 About"])
with tab1:
    st.write("Mean Score:", data["Score"].mean())
with tab2:
    st.markdown("This tab demonstrates how to use **tabs** in Streamlit.")

# Buttons
if st.button("Click Me!"):
    st.balloons()
    st.success("You clicked the button!")

st.divider()

# ========================================================================
# 7️⃣ FILE UPLOAD & DOWNLOAD
# ========================================================================
st.header("6️⃣ File Upload & Download")

uploaded = st.file_uploader("Upload a CSV file", type=['csv'])
if uploaded:
    df_upload = pd.read_csv(uploaded)
    st.success(f"File uploaded! {df_upload.shape[0]} rows × {df_upload.shape[1]} columns")
    st.dataframe(df_upload.head())

    # Download sample
    csv = df_upload.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Processed File", data=csv, file_name="processed.csv", mime='text/csv')
else:
    st.info("👆 Upload a CSV file to preview here.")

st.divider()

# ========================================================================
# 8️⃣ STATUS MESSAGES
# ========================================================================
st.header("7️⃣ Status Messages and Feedback")

st.success("✅ Success message example")
st.info("ℹ️ Informational message")
st.warning("⚠️ Warning message")
st.error("❌ Error message")
try:
    1 / 0
except Exception as e:
    st.exception(e)

st.divider()

# ========================================================================
# 9️⃣ SESSION STATE (Remembering Values)
# ========================================================================
st.header("8️⃣ Session State Example")

if "counter" not in st.session_state:
    st.session_state.counter = 0

col1, col2 = st.columns(2)
if col1.button("➕ Increment"):
    st.session_state.counter += 1
if col2.button("🔄 Reset"):
    st.session_state.counter = 0

st.metric("Counter", st.session_state.counter)

st.divider()

# ========================================================================
# 🔚 FOOTER
# ========================================================================
st.markdown("""
---
### 🏁 Summary of Concepts
✅ Text & markdown  
✅ Input widgets  
✅ Data display  
✅ Visualizations  
✅ Layout & sidebar  
✅ File handling  
✅ Session state & interactivity  

Built with ❤️ using **Streamlit**  
📘 Official docs: [https://docs.streamlit.io](https://docs.streamlit.io)
""")
