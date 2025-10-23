
# 🎓 Gradio for Data Science — Complete Hands-On Guide  
*(With Custom CSS and End-to-End Example Project)*

---

## 🧭 Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Creating Your First App](#creating-your-first-app)
4. [Input Components](#input-components)
5. [Output Components](#output-components)
6. [Image and Media Processing](#image-and-media-processing)
7. [Blocks Layout System](#blocks-layout-system)
8. [Chatbots and Conversational UIs](#chatbots-and-conversational-uis)
9. [Interactive Dashboards](#interactive-dashboards)
10. [ML Model Demos](#ml-model-demos)
11. [Custom CSS and Theming](#custom-css-and-theming)
12. [End-to-End Project: Image Classifier App](#end-to-end-project-image-classifier-app)
13. [Sharing and Hosting](#sharing-and-hosting)
14. [Best Practices for Data Science Projects](#best-practices-for-data-science-projects)
15. [References](#references)

---

## 🌟 Introduction

**Gradio** is an open-source Python library for creating **interactive web apps**, **ML demos**, and **data science dashboards** directly from Python functions.

It’s perfect for:
- 🚀 Rapid prototyping of ML models
- 🎛️ Interactive visualization of data
- 🧠 Deploying and sharing AI demos easily

---

## ⚙️ Installation

Install Gradio:
```bash
pip install gradio
````

Run this to confirm:

```python
import gradio as gr
print(gr.__version__)
```

---

## 🧩 Creating Your First App

```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Enter your name"),
    outputs="text",
    title="👋 Greeting App"
)

demo.launch(share=True)
```

✅ **Explanation:**

* `fn`: the Python function to run
* `inputs` & `outputs`: UI elements
* `share=True`: creates a **temporary public link**

---

## 🎛️ Input Components

| Component        | Description          | Example                                 |
| ---------------- | -------------------- | --------------------------------------- |
| `gr.Textbox()`   | Text input           | `gr.Textbox(label="Comment")`           |
| `gr.Number()`    | Numeric input        | `gr.Number(label="Age")`                |
| `gr.Slider()`    | Numeric slider       | `gr.Slider(0,100,50,label="Threshold")` |
| `gr.Checkbox()`  | Boolean              | `gr.Checkbox(label="Normalize Data")`   |
| `gr.Radio()`     | Single choice        | `gr.Radio(["SVM","LR","RF"])`           |
| `gr.Dropdown()`  | Select from list     | `gr.Dropdown(["A","B","C"])`            |
| `gr.File()`      | Upload a file        | `gr.File(label="Upload CSV")`           |
| `gr.Image()`     | Upload or draw image | `gr.Image(shape=(224,224))`             |
| `gr.Dataframe()` | Editable table       | `gr.Dataframe()`                        |

💡 Combine inputs:

```python
inputs = [gr.Textbox(), gr.Slider(0,10)]
```

---

## 📤 Output Components

| Component        | Description           | Example                    |
| ---------------- | --------------------- | -------------------------- |
| `gr.Textbox()`   | Text output           | `"Result: OK"`             |
| `gr.Number()`    | Numeric output        | `42`                       |
| `gr.Dataframe()` | Table output          | `pd.DataFrame()`           |
| `gr.Plot()`      | Graph output          | `matplotlib.figure.Figure` |
| `gr.Image()`     | Image output          | `numpy.ndarray`            |
| `gr.JSON()`      | JSON structure        | `{"a":1,"b":2}`            |
| `gr.HTML()`      | Render HTML           | `"<h3>Done</h3>"`          |
| `gr.Label()`     | Classification result | `{"Cat":0.9,"Dog":0.1}`    |

---

## 🖼️ Image and Media Processing

```python
from PIL import Image, ImageEnhance
import numpy as np
import gradio as gr

def adjust_brightness(image, intensity):
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(intensity))

gr.Interface(
    fn=adjust_brightness,
    inputs=[gr.Image(), gr.Slider(0.5,2.0,1.0)],
    outputs=gr.Image()
).launch()
```

✅ *Ideal for computer vision and preprocessing demos.*

---

## 🧱 Blocks Layout System

Gradio’s **Blocks** API allows advanced layouts with tabs, rows, and columns.

```python
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Sentiment Analysis Dashboard")
    text_input = gr.Textbox(label="Enter Review")
    analyze_btn = gr.Button("Analyze")
    output = gr.Label(label="Sentiment")
    analyze_btn.click(fn=predict_sentiment, inputs=text_input, outputs=output)
demo.launch()
```

* `gr.Row()` — Arrange horizontally
* `gr.Column()` — Stack vertically
* `gr.Tab()` — Separate workflows

---

## 💬 Chatbots and Conversational UIs

```python
def chat(message, history):
    history = history or []
    reply = f"🤖 Bot: I heard '{message}'"
    history.append((message, reply))
    return "", history

with gr.Blocks() as chat_ui:
    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox()
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
chat_ui.launch()
```

✅ Great for conversational AI and NLP demos.

---

## 📈 Interactive Dashboards

```python
import pandas as pd, matplotlib.pyplot as plt, gradio as gr

def plot_chart(feature):
    df = pd.DataFrame({"x":range(10), feature:[i**2 for i in range(10)]})
    fig, ax = plt.subplots()
    ax.plot(df["x"], df[feature])
    return fig

gr.Interface(fn=plot_chart,
             inputs=gr.Dropdown(["x^2"], label="Feature"),
             outputs=gr.Plot()).launch()
```

✅ Build simple or complex data exploration tools.

---

## 🤖 ML Model Demos

```python
import joblib, gradio as gr

model = joblib.load("model.pkl")

def predict(age, income):
    pred = model.predict([[age, income]])[0]
    return {"Approved": float(pred), "Rejected": 1-float(pred)}

gr.Interface(fn=predict,
             inputs=[gr.Number(label="Age"), gr.Number(label="Income")],
             outputs=gr.Label(),
             title="Loan Predictor").launch(share=True)
```

💡 Works seamlessly with **scikit-learn**, **PyTorch**, and **TensorFlow**.

---

## 🎨 Custom CSS and Theming

You can apply **custom CSS** or **themes** to personalize your Gradio apps.

### 🧱 Custom CSS Example

```python
custom_css = """
body { background-color: #f9fafb; font-family: 'Inter', sans-serif; }
h1 { color: #2563eb; text-align: center; }
.gr-button { background-color: #2563eb !important; color: white !important; border-radius: 8px !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎨 Custom Styled App")
    name = gr.Textbox(label="Name")
    btn = gr.Button("Greet Me")
    output = gr.Textbox(label="Greeting")

    def greet(name): return f"Hello, {name}!"
    btn.click(greet, name, output)
demo.launch()
```

### 🌈 Themes Example

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌈 Soft Theme Example")
    gr.Textbox(label="Write something...")
demo.launch()
```

Themes available:

* `gr.themes.Soft()`
* `gr.themes.Glass()`
* `gr.themes.Monochrome()`
* `gr.themes.SolarizedLight()`

---

## 🧠 End-to-End Project: Image Classifier App

This project demonstrates **everything**: input, output, CSS, theming, interactivity, and deployment.

### 📂 Project Overview

A small CNN image classifier demo that predicts **Dog** or **Cat** using a pre-trained model.

---

### 🧩 Code

```python
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("cat_dog_model.h5")
class_names = ["Cat", "Dog"]

# Prediction Function
def classify_image(img):
    img = Image.fromarray(img).resize((150,150))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    preds = model.predict(img_array)
    label = class_names[int(preds[0][0] > 0.5)]
    confidence = float(preds[0][0]) if label=="Dog" else 1-float(preds[0][0])
    return {label: confidence}

# Custom Styling
custom_css = """
body { background: linear-gradient(135deg,#eef2ff,#e0f2fe); font-family: 'Poppins'; }
h1 { color:#0f172a; text-align:center; font-size:28px; }
.gr-button { background-color:#1d4ed8 !important; color:white !important; border-radius:10px; }
"""

# UI Layout
with gr.Blocks(theme=gr.themes.Glass(), css=custom_css) as demo:
    gr.Markdown("# 🐾 Cat vs Dog Classifier")
    with gr.Row():
        img_input = gr.Image(label="Upload an Image", shape=(150,150))
        result = gr.Label(label="Prediction")
    predict_btn = gr.Button("Predict 🧠")
    predict_btn.click(classify_image, inputs=img_input, outputs=result)

demo.launch(share=True)
```

### 🔍 Features Used:

✅ `gr.Image()` input
✅ `gr.Label()` for model output
✅ `Blocks` layout with `Row()`
✅ Custom CSS for background
✅ `Glass()` theme
✅ Public sharing link

---

## 🌐 Sharing and Hosting

| Option                  | Description                | Command                                                             |
| ----------------------- | -------------------------- | ------------------------------------------------------------------- |
| **Local run**           | Launch app on your machine | `python app.py`                                                     |
| **Public share**        | Temporary link (72 hrs)    | `demo.launch(share=True)`                                           |
| **Hugging Face Spaces** | Free permanent hosting     | Upload repo to [Hugging Face Spaces](https://huggingface.co/spaces) |
| **Custom deploy**       | Docker / server hosting    | `gradio deploy`                                                     |

---

## 🧠 Best Practices for Data Science Projects

✅ Use **Interface** for one-step inference apps
✅ Use **Blocks** for dashboards or multiple features
✅ Label your widgets properly
✅ Add sample examples for users
✅ Use **CSS + themes** for professional UI
✅ Cache model weights for faster inference
✅ Include **requirements.txt** for deployment

---

## 📚 References

* [Gradio Official Docs](https://gradio.app/docs/)
* [Hugging Face Spaces](https://huggingface.co/spaces)
* [Gradio GitHub](https://github.com/gradio-app/gradio)
* [Gradio Examples](https://gradio.app/guides/)

---

> ✨ *Gradio allows data scientists to build, deploy, and share interactive machine learning demos within minutes — no web development required.*

```

---

Would you like me to now make this `.md` file **downloadable** (e.g., as a `.zip` or `.md` attachment) so you can directly upload it to GitHub or your course notes?
```
