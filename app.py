import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_model():
    return joblib.load('random_forest_classifier.pkl')

def preprocess_and_predict(df, model):
    features = ['Depth', 'GR', 'SP', 'MCAL', 'DCAL', 'MI', 'RxoRt', 'RLL3', 'RILD', 'RHOB', 'RHOC', 'DPOR', 'MN', 'CNLS']
    df = df[features].dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    predictions = model.predict(scaled_features)
    
    df['Lithology Prediction'] = predictions
    return df

def main():
    st.title("Well Log Lithology Prediction App")
    st.write("Upload well log data to predict lithology classes using a trained Random Forest model.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())
        
        model = load_model()
        result_df = preprocess_and_predict(df, model)
        
        st.write("### Lithology Predictions:")
        st.dataframe(result_df.head())
        
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
        
if __name__ == "__main__":
    main()