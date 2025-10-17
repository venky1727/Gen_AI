import streamlit as st
import pandas as pd
import os
import numpy as np
import json
from dotenv import load_dotenv
from predict_ai import predict_from_vertex
from utils.gcs_helper import download_blob_as_df
 
# Load environment variables
load_dotenv()
 
# Title
st.title("Customer 360 Trade Finance Risk Prediction")
 
# Input: GCS path or CSV upload
gcs_path = st.text_input("Enter GCS path (e.g. gs://your-bucket/input.csv)")
uploaded_file = st.file_uploader("...or upload a CSV file", type=["csv"])
 
# Load data
df = None
if gcs_path:
    try:
        df = download_blob_as_df(gcs_path)
        st.success("File loaded from GCS successfully!")
    except Exception as e:
        st.error(f"Failed to load from GCS: {e}") # Added error details
 
elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully!")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}") # Added error details
 
# Run prediction
if df is not None:
    st.subheader("Uploaded Data")
    st.dataframe(df)
 
    # Ensure correct types
    columns_to_cast_as_string = [
        'Years_in_Operation', 'Annual_Revenue', 'Customer_Id', 'Company_ID', 'Profitability_Status',
        'Requested_Guarantee', 'Total_Requested_Guarantee', 'Max_Requested_Guarantee',
        'On_Time_Payment_Rate', 'Avg_Days_Late', 'Product_Mix', 'Country', 'Counterparty_Country','Profit_Margin','Debt_to_Equity','Credit_Score','Previous_BGs',
        'Total_BG_Value','Default_Count','Transaction_Value','Requested_Guarantee','Total_Requested_Guarantee','Max_Requested_Guarantee','On_Time_Payment_Rate','Avg_Days_Late','Cash_Flow_Gap_Days'
    ]
 
    for col in columns_to_cast_as_string:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
 
    # Replace NaN with None
    df = df.where(pd.notnull(df), None)
 
    if st.button("Predict"):
        try:
            # Prepare instances for batch prediction
            instances = df.to_dict(orient="records")
 
            # Call Vertex AI
            response = predict_from_vertex(
                endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
                project=os.getenv("PROJECT_ID"),
                location=os.getenv("REGION"),
                instance_list=instances
            )
 
            st.subheader("Prediction Results")
 
            if not response or not hasattr(response, 'predictions') or not response.predictions:
                st.warning("No predictions received from the model.")
            else:
                for i, prediction in enumerate(response.predictions):
                    predicted_class = "Unknown"
                    confidence = 0.0
 
                    try:
                        # Try Format 1: displayName + confidence
                        if 'displayName' in prediction and 'confidence' in prediction:
                            predicted_class = prediction['displayName']
                            confidence_val = prediction['confidence']
                            if isinstance(confidence_val, (int, float)):
                                confidence = float(confidence_val)
                            else:
                                st.warning(f"Record {i+1}: Invalid confidence format for displayName format.")
 
                        # Try Format 2: classes + scores
                        elif 'classes' in prediction and 'scores' in prediction:
                            if prediction['scores'] and prediction['classes']:
                                scores = np.array(prediction['scores'])
                                classes = prediction['classes']
                                if scores.size > 0 and len(classes) == scores.size:
                                    idx = np.argmax(scores)
                                    predicted_class = classes[idx]
                                    confidence_val = scores[idx]
                                    if isinstance(confidence_val, (int, float)):
                                        confidence = float(confidence_val)
                                    else:
                                        st.warning(f"Record {i+1}: Invalid confidence format for classes/scores format.")
                                else:
                                    st.warning(f"Record {i+1}: Mismatch or empty scores/classes in classes/scores format.")
                            else:
                                st.warning(f"Record {i+1}: Empty scores or classes found.")
                        else:
                            st.warning(f"Record {i+1}: Unrecognized prediction format.")
 
                        # Display result
                        if isinstance(confidence, (int, float)):
                            st.write(f"**Record {i+1}:** Predicted Class = {predicted_class}, Confidence = {round(confidence * 100, 2)}%")
                        else:
                            st.write(f"**Record {i+1}:** Predicted Class = {predicted_class}, Confidence = N/A (Invalid format)")
 
                    except Exception as parse_err:
                        st.error(f"Prediction parsing failed for record {i+1}: {parse_err}")
                        # Optional: log raw data for debugging
                        # st.write(f"**Record {i+1}:** Raw prediction data: {prediction}")
 
        except Exception as e:
            st.error(f"Prediction failed: {e}")