import joblib
import pandas as pd

# 1. EXACT ABSOLUTE PATHS TO YOUR SAVED FILES
# (Double check these paths match your computer!)
MODEL_PATH = r"C:\Users\Ahmad Reza\Freight Price Prediction\invoice_flagging\models\predict_flag_invoice.pkl"
SCALER_PATH = r"C:\Users\Ahmad Reza\Freight Price Prediction\invoice_flagging\models\scaler.pkl"

def load_pipeline():
    """Loads both the trained model and the scaler (the translator)."""
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
        
    with open(SCALER_PATH, "rb") as f:
        scaler = joblib.load(f)
        
    return model, scaler

def predict_invoice_risk(input_data):
    """
    Predicts if an invoice is Normal (0) or Risky (1).
    Requires the 5 exact features used during training.
    """
    model, scaler = load_pipeline()
    
    # 1. Convert input to DataFrame (ensuring correct column order)
    features = [
        'invoice_quantity', 
        'invoice_dollars', 
        'Freight', 
        'total_item_quantity', 
        'total_item_dollars'
    ]
    input_df = pd.DataFrame(input_data)[features]
    
    # 2. TRANSLATE: Scale the raw numbers just like we did in training
    input_scaled = scaler.transform(input_df)
    
    # 3. PREDICT: Get the 0 or 1 predictions
    predictions = model.predict(input_scaled)
    
    # 4. Make the output human-readable for the finance team
    result_df = input_df.copy()
    result_df['Risk_Code'] = predictions
    result_df['Status'] = result_df['Risk_Code'].map({0: 'Normal', 1: 'RISKY (Review)'})
    
    return result_df

# ==========================================
# THIS IS THE NEW PART STREAMLIT NEEDS
# ==========================================
def predict_invoice_flag(input_data):
    """
    Bridge function for Streamlit. 
    Calls your original function and formats the output for the web app.
    """
    # 1. Run your perfectly working original function
    df_result = predict_invoice_risk(input_data)
    
    # 2. Extract just the 0 or 1 prediction and package it in a dictionary for app.py
    return {'Predicted_flag': df_result['Risk_Code'].values}
# ==========================================

if __name__ == "__main__":
    # Let's test two fake invoices. 
    # Invoice 1 looks perfect (items match the invoice).
    # Invoice 2 looks highly suspicious (Invoice bills for $5000, but items only equal $1000!)
    sample_invoices = {
        'invoice_quantity': [100, 50],
        'invoice_dollars': [1000.00, 5000.00],
        'Freight': [50.00, 800.00],
        'total_item_quantity': [100, 50],
        'total_item_dollars': [1000.00, 1000.00] 
    }
    
    print("\nLoading model and predicting risk...")
    results = predict_invoice_risk(sample_invoices)
    
    print("\n--- Invoice Risk Predictions ---")
    # We will just print a few columns to keep the screen clean
    print(results[['invoice_dollars', 'total_item_dollars', 'Freight', 'Status']])