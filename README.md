# 🎬 Anatomy of a Blockbuster  
### *What Makes a Movie a Hit?*

This project investigates what factors contribute to a movie becoming a **blockbuster**.  
Using the Kaggle *Movies Dataset*, we analyze how **genre**, **budget**, **runtime**, **release date**, **cast & crew**, and **production companies** influence **box office revenue and audience reception**.

An interactive **Streamlit dashboard** is included for exploring these insights.
---
📥 **Download the Dataset:**  
The Movies Dataset is available on Kaggle:  
👉 https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

---

## 🚀 Features

### 🔍 Data Exploration
- Distribution of movies by:
  - Genre  
  - Country  
  - Production company  
  - Runtime  
  - Release year  
- Summary statistics for budget, revenue, and ROI.

### 📈 Advanced Analytics
- Yearly revenue trends  
- Hit vs non-hit comparison  
- Correlation heatmap (budget, revenue, popularity, etc.)  
- Genre performance  
- Actor/Director contribution analysis  
- Company-level success rates  
- Geographic distribution using PyDeck + GeoJSON  

### 🎛️ Interactive Streamlit App
- Clean dashboard UI  
- Modularized codebase (`src/` directory)  
- Multiple visualization types  
- Efficient backend with preprocessed PKL files  

---

## 📂 Project Structure
```
movie-blockbuster-analysis/
│
├── data/
│ ├── credits_lead.pkl
│ ├── credits.csv
│ ├── movies_clean_full.pkl
│ ├── movies_metadata.csv
│ ├── movies_metadata.pkl
│ └── world_countries.geojson
│
├── src/
│ ├── init.py
│ ├── analysis.py
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── pydeck_utils.py
│ └── visualization.py
│
├── app.py
├── requirements.txt
└── README.md
---
```
## 🗂️ Module Descriptions

| File | Description |
|------|-------------|
| `data_loader.py` | Loads raw and cleaned movie datasets. |
| `preprocess.py` | Cleans data, handles missing values, creates ROI & blockbuster features. |
| `analysis.py` | High-level statistical summaries and backend analytics. |
| `visualization.py` | Generates charts and visualizations (Plotly, Matplotlib, PyDeck). |
| `pydeck_utils.py` | Utilities for global geographic maps. |
| `app.py` | Streamlit application that integrates all components. |
| `data/` | Raw CSV files, cleaned PKLs, and geographic data. |

---

## 🧾 Dataset

**The Movies Dataset**  
🔗 https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Includes:
- 45k+ movies  
- Cast & crew metadata  
- Budgets and worldwide revenue  
- Genres, keywords, and production companies  
- Ratings and user scores  

---

## ⭐ Definition: What Is a “Blockbuster”?

In this project, a **blockbuster** is defined using multiple criteria such as:

- **ROI (Return on Investment)**  
- **Absolute revenue thresholds**  
- **Industry-standard definitions**  

A new column `is_blockbuster` is created during preprocessing.

---
## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**, **NumPy**
- **Matplotlib**, **Plotly**
- **PyDeck** + **GeoJSON**
- **Streamlit**
- **Pickle (PKL)** for optimized loading
---
## ▶️ Running the Project

### 1. Clone and installs
```bash
git clone https://github.com/Behradsadeghi/Anatomy-of-a-Blockbuster.git
cd Anatomy-of-a-Blockbuster
```
### Create a virtual environment (recommended)
```
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```
### Install dependencies
```
pip install -r requirements.txt
```
### Launch Streamlit app
```
streamlit run app.py
```

