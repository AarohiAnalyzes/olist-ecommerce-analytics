# 🛒 Olist Brazilian E-Commerce Analysis

Interactive end-to-end business analysis of the Brazilian Olist marketplace dataset, covering **sales performance**, **customer geography**, and **product/seller dynamics** for January 2017 - August 2018.

🔗 **Live Dashboard:** [olist-ecommerce-analytics-by-aarohi.streamlit.app](https://olist-ecommerce-analytics-by-aarohi.streamlit.app)

---

## 📊 Project Overview

This project answers three core business questions:

1. **Sales Performance**: How does monthly revenue evolve? Are there seasonal patterns?
2. **Customer Geography**: Where are customers concentrated? Which states drive revenue?
3. **Product & Seller Insights**: Which product categories dominate revenue? How do seller strategies differ? Does price drive satisfaction?

---

## 🎯 Deliverables

- ✅ **Jupyter Notebook** (`notebooks/`): full analysis with data cleaning, EDA, and statistical modelling
- ✅ **Interactive Streamlit Dashboard** (`dashboard/app.py`): 5-page interactive view of findings, deployed live
- ✅ **Documentation** (`docs/`): written insights and methodology
- ✅ **Schema & ERD** (`assets/`): data model documentation

---

## 📁 Project Structure
- `assets/` - Schema diagrams and notes
- `docs/` - Project documentation
- `notebooks/` - Analysis notebooks (coming soon)
- `data/` - Data source information

---

## 🛠️ Tech Stack

- **Python** - Pandas, NumPy
- **Streamlit** - interactive web framework
- **Plotly** - interactive visualizations
- **Matplotlib & Seaborn** - static visualizations in notebooks
- **SQL-style joins** via Pandas `merge()` operations across 6 relational tables

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/AarohiAnalyzes/olist-ecommerce-analytics.git
cd olist-ecommerce-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📊 Dashboard Pages

The interactive dashboard includes 5 main sections:

| Page | Content |
|---|---|
| 🏠 **Overview** | Top-level KPIs and project context |
| 💰 **Sales Performance** | Monthly revenue trend, seasonality, best/worst months |
| 🌎 **Customer Geography** | State-level customer & revenue distribution |
| 📦 **Product & Seller Insights** | Category revenue, review scores, seller strategies, price vs. satisfaction |
| 🔑 **Key Insights** | Synthesized findings and business recommendations |

---

## 🔍 Key Findings

✔ **~R$ 13M total revenue** generated across delivered orders (Jan 2017 - Aug 2018)  
✔ **Strong revenue seasonality** with peak in November 2017 (Black Friday effect)  
✔ **Southeast Brazil** drives the majority of customers and revenue  
✔ Top sellers split between **premium** and **volume** strategies: both work  
✔ **Price vs. Review Score correlation ≈ 0** — price doesn't drive satisfaction; logistics and quality do  

---

## 🔗 Data Source

**Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
**Platform:** Kaggle

---

## 👤 Author

**Aarohi Mistry**  
M.Sc. Data Science — Università degli Studi di Milano-Bicocca

[LinkedIn](https://linkedin.com/in/aarohi-mistry-715713219) · [GitHub](https://github.com/AarohiAnalyzes)
