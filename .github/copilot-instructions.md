# GitHub Copilot Instructions

This repository contains a machine learning web application for optimizing fermentation media composition to predict **Media Cost**, **Enzyme Activity (U/mL)**, and **Media Cost Efficiency** using scikit-learn models served via a Gradio interface.

## Project Context

- **Language:** Python 3.9+
- **Key libraries:** Gradio, scikit-learn, pandas, NumPy
- **Entry point:** `app.py`
- **Pre-trained models:** `final_cost_model.pkl` (Linear Regression) and `final_ea_model.pkl` (Random Forest)
- **Input features:** CSL (%v/v), Molasses (%v/v), WCO (%v/v)
- **Outputs:** Predicted Media Cost, Enzyme Activity, and Cost Efficiency

## GitHub Copilot Model Availability

> **Note:** `claude-opus-4.6` (Claude Opus) is no longer available in GitHub Copilot.
>
> Please use one of the currently available Claude models instead:
> - **`claude-3.5-sonnet`** — Recommended for most tasks (fast, capable, free-tier eligible)
> - **`claude-3.7-sonnet`** — Latest Claude model with enhanced reasoning
>
> To switch models in VS Code, open Copilot Chat and click the model selector at the bottom of the chat panel, then choose an available Claude model.

## Coding Conventions

- Follow PEP 8 style guidelines
- Use pandas DataFrames for model input (column names must match training features exactly)
- Handle division-by-zero cases explicitly (see `predict_medium_properties` in `app.py`)
- Keep model loading separate from prediction logic
