
# 🎓 Streamlit for Data Science — Complete Guide

Streamlit is a powerful open-source Python framework that allows data scientists to **turn their scripts and models into interactive web apps** with minimal effort.  
This guide covers **all major features** of Streamlit with examples, best practices, and usage tips for data science workflows.

---

## 🧭 Table of Contents
1. [Introduction](#introduction)
2. [App Configuration](#app-configuration)
3. [Text and Display Elements](#text-and-display-elements)
4. [Input Widgets](#input-widgets)
5. [Data Display Tools](#data-display-tools)
6. [Charts and Visualizations](#charts-and-visualizations)
7. [Layout and Interactivity](#layout-and-interactivity)
8. [File Handling](#file-handling)
9. [Feedback and Status Messages](#feedback-and-status-messages)
10. [Session State](#session-state)
11. [Caching and Performance](#caching-and-performance)
12. [Deploying a Streamlit App](#deploying-a-streamlit-app)
13. [Best Practices for Data Science Projects](#best-practices-for-data-science-projects)

---

## 🧩 Introduction

**Streamlit** bridges the gap between data science and web development by enabling:
- Instant visualization of data and models.
- Interactive parameter tuning.
- Real-time feedback for experimentation.
- Web deployment without HTML, CSS, or JavaScript.

📘 Install Streamlit:
```bash
pip install streamlit
````

Run an app:

```bash
streamlit run app.py
```

---

## ⚙️ App Configuration

```python
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

**Key Parameters:**

* `page_title`: Text shown on the browser tab.
* `page_icon`: Emoji or icon.
* `layout`: `'wide'` or `'centered'`.
* `initial_sidebar_state`: `'expanded'` or `'collapsed'`.

💡 *Tip:* Always configure your page at the start of your script.

---

## 📝 Text and Display Elements

| Function         | Description                              | Example                                                     |
| ---------------- | ---------------------------------------- | ----------------------------------------------------------- |
| `st.title()`     | Large heading for the app                | `st.title("Data Science Dashboard")`                        |
| `st.header()`    | Section title                            | `st.header("Exploratory Analysis")`                         |
| `st.subheader()` | Subsection title                         | `st.subheader("Feature Distribution")`                      |
| `st.text()`      | Displays plain text                      | `st.text("Simple text output")`                             |
| `st.markdown()`  | Renders Markdown formatting              | `st.markdown("**Bold Text** and *Italics*")`                |
| `st.write()`     | Auto-detects data type (text, df, chart) | `st.write(df.describe())`                                   |
| `st.code()`      | Displays code blocks                     | `st.code("for i in range(5): print(i)", language="python")` |
| `st.latex()`     | Renders LaTeX math formulas              | `st.latex(r"E=mc^2")`                                       |

📊 *Usage Example:*

```python
st.header("Model Equation")
st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon")
```

---

## 🎛️ Input Widgets

Streamlit’s widgets make apps interactive.

| Widget               | Function                   | Example                                                 |
| -------------------- | -------------------------- | ------------------------------------------------------- |
| `st.button()`        | Simple clickable button    | `if st.button("Run Model"): ...`                        |
| `st.checkbox()`      | True/False toggle          | `st.checkbox("Show data")`                              |
| `st.radio()`         | Choose one option          | `st.radio("Select Model", ["LR", "RF", "SVM"])`         |
| `st.selectbox()`     | Dropdown list              | `st.selectbox("Select Feature", df.columns)`            |
| `st.multiselect()`   | Select multiple options    | `st.multiselect("Select Columns", df.columns)`          |
| `st.slider()`        | Select numeric value range | `st.slider("Learning Rate", 0.0, 1.0, 0.1)`             |
| `st.number_input()`  | Enter numeric value        | `st.number_input("Enter k", min_value=1, max_value=10)` |
| `st.text_input()`    | Input text                 | `st.text_input("Dataset name")`                         |
| `st.date_input()`    | Select date                | `st.date_input("Pick a date")`                          |
| `st.file_uploader()` | Upload file                | `st.file_uploader("Upload CSV", type=['csv'])`          |

💡 *Tip:* Use input widgets to create parameterized models or dashboards (e.g., model tuning sliders).

---

## 📊 Data Display Tools

| Function         | Description        | Example                               |
| ---------------- | ------------------ | ------------------------------------- |
| `st.dataframe()` | Interactive table  | `st.dataframe(df.head())`             |
| `st.table()`     | Static table       | `st.table(df.describe())`             |
| `st.json()`      | Pretty-prints JSON | `st.json(model_params)`               |
| `st.metric()`    | Displays KPIs      | `st.metric("Accuracy", "95%", "+2%")` |

📘 *Use Case:*
Display summary statistics of a dataset:

```python
st.dataframe(df.describe())
st.metric("Rows", len(df))
st.metric("Columns", df.shape[1])
```

---

## 📈 Charts and Visualizations

Streamlit integrates seamlessly with **Matplotlib**, **Seaborn**, **Plotly**, and **Altair**.

### Built-in Charts

```python
st.line_chart(df)
st.bar_chart(df)
st.area_chart(df)
```

### Matplotlib Example

```python
fig, ax = plt.subplots()
ax.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
st.pyplot(fig)
```

### Seaborn Example

```python
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
```

### Plotly Example

```python
import plotly.express as px
fig = px.scatter(df, x='Feature1', y='Feature2', color='Target')
st.plotly_chart(fig)
```

💡 *For data scientists:* Combine model predictions and metrics in visual dashboards.

---

## 🧩 Layout and Interactivity

Streamlit supports flexible page layouts.

### Columns

```python
col1, col2 = st.columns(2)
col1.write("Left Column")
col2.write("Right Column")
```

### Tabs

```python
tab1, tab2 = st.tabs(["📈 Data", "⚙️ Model"])
with tab1:
    st.dataframe(df)
with tab2:
    st.write("Model details here")
```

### Sidebar

```python
with st.sidebar:
    st.header("Settings")
    param = st.slider("Parameter value", 0, 100, 50)
```

💡 *For dashboards:* Place filters or controls in the sidebar for a clean layout.

---

## 📂 File Handling

### Upload Files

```python
uploaded = st.file_uploader("Upload a CSV file", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
```

### Download Files

```python
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Processed File", data=csv, file_name="output.csv", mime="text/csv")
```

💡 Use upload/download for preprocessed datasets, model outputs, or user reports.

---

## ⚠️ Feedback and Status Messages

| Function          | Purpose                 |
| ----------------- | ----------------------- |
| `st.success()`    | Green success message   |
| `st.info()`       | Blue informational box  |
| `st.warning()`    | Yellow warning          |
| `st.error()`      | Red error message       |
| `st.exception(e)` | Prints Python traceback |

### Example:

```python
try:
    st.success("Data loaded successfully!")
    result = model.predict(X)
except Exception as e:
    st.exception(e)
```

---

## 🔁 Session State

Streamlit runs top-to-bottom on every change, so we use **`st.session_state`** to store persistent variables.

```python
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Increment"):
    st.session_state.counter += 1

st.metric("Counter", st.session_state.counter)
```

💡 *Use Case:* Save user inputs, model parameters, or intermediate results.

---

## ⚡ Caching and Performance

For long computations or model loading, use caching to avoid re-execution.

```python
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))
```

💡 *Difference:*

* `@st.cache_data`: caches outputs of pure functions (data processing).
* `@st.cache_resource`: caches objects like ML models or database connections.

---

## 🌐 Deploying a Streamlit App

### Option 1: Streamlit Cloud

* Upload your code to GitHub.
* Visit [https://share.streamlit.io](https://share.streamlit.io)
* Connect your repo → auto-deploy.

### Option 2: Local Deployment

```bash
streamlit run app.py
```

### Option 3: ngrok (for Google Colab)

```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```

---

## 📚 References

* [Streamlit Official Docs](https://docs.streamlit.io)
* [Awesome Streamlit Projects](https://github.com/MarcSkovMadsen/awesome-streamlit)
* [Streamlit Cheat Sheet (by Data Professor)](https://github.com/daniellewisDL/streamlit-cheat-sheet)

---

### 🏁 Summary

Streamlit enables **data scientists** to:

* Prototype faster 🧪
* Visualize interactively 📊
* Share models easily 🌐
* Create dashboards without web dev skills 💻

With just a few lines of Python, you can turn your analysis into an interactive, professional app!

---

```

---

Would you like me to add a **“Data Science–Focused Example Section”** (like “Build a Machine Learning Dashboard using Streamlit”) inside this `.md` file before exporting it? It would show real-world usage with code + explanations.
```
