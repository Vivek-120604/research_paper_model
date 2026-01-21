import gradio as gr
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Models ---
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

final_cost_model = load_model('final_cost_model.pkl')
final_ea_model = load_model('final_ea_model.pkl')

# --- 2. Define Prediction Function ---
def predict_medium_properties(csl, molasses, wco):
    input_df = pd.DataFrame({
        'A: CSL (%v/v)': [csl],
        'B: Molasses (%v/v)': [molasses],
        'C: WCO (%v/v)': [wco]
    })

    predicted_media_cost = final_cost_model.predict(input_df)[0]
    predicted_enzyme_activity = final_ea_model.predict(input_df)[0]

    if predicted_media_cost != 0:
        predicted_efficiency = predicted_enzyme_activity / predicted_media_cost
    else:
        predicted_efficiency = np.inf # Handle division by zero

    return predicted_media_cost, predicted_enzyme_activity, predicted_efficiency

# --- 3. Set up Gradio Interface ---
iface = gr.Interface(
    fn=predict_medium_properties,
    inputs=[
        gr.Slider(minimum=0.5, maximum=3.0, value=1.5, step=0.01, label='CSL (%v/v)'),
        gr.Slider(minimum=0.5, maximum=3.0, value=1.5, step=0.01, label='Molasses (%v/v)'),
        gr.Slider(minimum=0.25, maximum=1.25, value=0.625, step=0.005, label='WCO (%v/v)')
    ],
    outputs=[
        gr.Number(label='Predicted Media Cost', precision=4),
        gr.Number(label='Predicted Enzyme Activity (U/mL)', precision=2),
        gr.Number(label='Predicted Media Cost Efficiency', precision=2)
    ],
    title='Media Optimization Predictor',
    description='Enter the values for CSL, Molasses, and WCO to predict Media Cost, Enzyme Activity, and Media Cost Efficiency.'
)

# Launch the interface
iface.launch()
