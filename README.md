# Media Optimization Predictor

A machine learning web application for optimizing fermentation media composition to maximize enzyme activity and cost efficiency. This tool predicts **Media Cost**, **Enzyme Activity (U/mL)**, and **Media Cost Efficiency** based on the concentrations of three fermentation medium components: CSL, Molasses, and WCO.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Visualizations](#visualizations)
- [GitHub Copilot Usage](#github-copilot-usage)
- [Model Performance](#model-performance)

---

## Project Overview

This project supports research on fermentation process optimization. By varying the concentrations of three media components — **Corn Steep Liquor (CSL)**, **Molasses**, and **Waste Cooking Oil (WCO)** — researchers can evaluate the trade-offs between cost and enzyme production yield.

Two pre-trained scikit-learn models are deployed via a **Gradio** web interface:

| Model | Algorithm | Target |
|---|---|---|
| `final_cost_model.pkl` | Linear Regression | Media Cost |
| `final_ea_model.pkl` | Random Forest Regressor | Enzyme Activity (U/mL) |

Cost Efficiency is then derived as:

```
Media Cost Efficiency = Enzyme Activity / Media Cost
```

---

## Features

- Interactive Gradio web UI with sliders for easy experimentation
- Real-time prediction of media cost, enzyme activity, and cost efficiency
- Pre-trained models ready for inference — no training step required
- Graceful handling of edge cases (e.g., zero-cost division)

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| [Gradio](https://gradio.app/) | 5.50.0 | Web interface |
| [scikit-learn](https://scikit-learn.org/) | 1.6.1 | Machine learning models |
| [Pandas](https://pandas.pydata.org/) | 2.2.2 | Data manipulation |
| [NumPy](https://numpy.org/) | 2.0.2 | Numerical computation |

**Python 3.9+** is recommended.

---

## Project Structure

```
research_paper_model/
├── app.py                                         # Gradio web application
├── requirements.txt                               # Python dependencies
├── final_cost_model.pkl                           # Trained Linear Regression model (media cost)
├── final_ea_model.pkl                             # Trained Random Forest model (enzyme activity)
│
├── # --- Visualizations ---
├── A_vs_Cost_Efficiency.png                       # CSL vs Cost Efficiency
├── A_vs_Enzyme_Activity.png                       # CSL vs Enzyme Activity
├── B_vs_Cost_Efficiency.png                       # Molasses vs Cost Efficiency
├── B_vs_Enzyme_Activity.png                       # Molasses vs Enzyme Activity
├── C_vs_Cost_Efficiency.png                       # WCO vs Cost Efficiency
├── C_vs_Enzyme_Activity.png                       # WCO vs Enzyme Activity
├── enzyme_activity_actual_vs_predicted.png        # Enzyme activity parity plot
├── media_cost_actual_vs_predicted.png             # Media cost parity plot
├── Parity_Media_Cost_Efficiency_Indirect_ML.png   # Cost efficiency parity plot
├── efficiency_ml_failure.png                      # Direct ML efficiency failure analysis
│
├── # --- Analysis Data ---
├── ml_optimization_table.xlsx                     # ML model performance metrics
├── Predicted_vs_Actual_MediaCost_EnzymeActivity.xlsx  # Validation predictions
├── New Microsoft Excel Worksheet (2).xlsx         # Supplementary data
│
└── sample_data/                                   # Reference datasets
    ├── README.md
    ├── anscombe.json
    ├── california_housing_train.csv
    ├── california_housing_test.csv
    ├── mnist_train_small.csv
    └── mnist_test.csv
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vivek-120604/research_paper_model.git
cd research_paper_model
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Gradio application:

```bash
python app.py
```

The application will launch and display a local URL (default: `http://127.0.0.1:7860`). Open it in your browser to use the interactive interface.

### Inputs

| Slider | Description | Range | Default |
|---|---|---|---|
| **CSL (%v/v)** | Corn Steep Liquor concentration | 0.50 – 3.00 | 1.50 |
| **Molasses (%v/v)** | Molasses concentration | 0.50 – 3.00 | 1.50 |
| **WCO (%v/v)** | Waste Cooking Oil concentration | 0.25 – 1.25 | 0.625 |

### Outputs

| Output | Description |
|---|---|
| **Predicted Media Cost** | Estimated cost of the media formulation (4 decimal places) |
| **Predicted Enzyme Activity (U/mL)** | Estimated enzyme production level (2 decimal places) |
| **Predicted Media Cost Efficiency** | Enzyme Activity ÷ Media Cost (2 decimal places) |

---

## Model Details

### Inputs (Features)

All three models share the same input feature set:

- `A: CSL (%v/v)` — Corn Steep Liquor
- `B: Molasses (%v/v)` — Molasses
- `C: WCO (%v/v)` — Waste Cooking Oil

### Cost Model — Linear Regression

A scikit-learn `LinearRegression` model trained to predict media cost from the three ingredient concentrations. The high R² (≈ 0.9963) reflects that media cost has a near-linear relationship with ingredient volumes.

### Enzyme Activity Model — Random Forest Regressor

A scikit-learn `RandomForestRegressor` model trained to capture the non-linear relationship between media composition and enzyme activity. Random Forest outperforms Linear Regression for this target (R² ≈ 0.84 vs. 0.61).

### Efficiency Calculation

Efficiency is computed directly in `app.py` rather than predicted by a standalone model, because indirect computation (Cost and Activity predicted separately) outperforms a direct ML approach for this metric:

```python
efficiency = enzyme_activity / media_cost
```

A direct ML approach to predict efficiency was evaluated and found to underperform (see `efficiency_ml_failure.png`).

---

## Visualizations

The repository includes the following analysis plots:

| File | Description |
|---|---|
| `A_vs_Enzyme_Activity.png` | Effect of CSL concentration on enzyme activity |
| `A_vs_Cost_Efficiency.png` | Effect of CSL concentration on cost efficiency |
| `B_vs_Enzyme_Activity.png` | Effect of Molasses concentration on enzyme activity |
| `B_vs_Cost_Efficiency.png` | Effect of Molasses concentration on cost efficiency |
| `C_vs_Enzyme_Activity.png` | Effect of WCO concentration on enzyme activity |
| `C_vs_Cost_Efficiency.png` | Effect of WCO concentration on cost efficiency |
| `enzyme_activity_actual_vs_predicted.png` | Actual vs. predicted enzyme activity (parity plot) |
| `media_cost_actual_vs_predicted.png` | Actual vs. predicted media cost (parity plot) |
| `Parity_Media_Cost_Efficiency_Indirect_ML.png` | Actual vs. predicted cost efficiency via indirect ML |
| `efficiency_ml_failure.png` | Analysis of why direct efficiency prediction underperforms |

---

## GitHub Copilot Usage

This repository is configured for use with [GitHub Copilot](https://github.com/features/copilot). Project-specific instructions for Copilot are stored in [`.github/copilot-instructions.md`](.github/copilot-instructions.md).

> **Model Availability Note:** `claude-opus-4.6` (Claude Opus) is no longer available through GitHub Copilot. Use a currently available Claude model instead, such as **`claude-3.5-sonnet`** or **`claude-3.7-sonnet`**. To switch, open Copilot Chat in VS Code and select a different model from the model picker.

---

## Model Performance

Summary of model evaluation metrics (from `ml_optimization_table.xlsx`):

| Target | Algorithm | R² | RMSE |
|---|---|---|---|
| Media Cost | Linear Regression | 0.9963 | ~0.0088 |
| Enzyme Activity | Random Forest | 0.8416 | — |
| Enzyme Activity | Linear Regression | 0.6134 | — |

The indirect approach (predict cost and activity separately, then divide) is used for cost efficiency because it significantly outperforms directly predicting efficiency with a machine learning model.
