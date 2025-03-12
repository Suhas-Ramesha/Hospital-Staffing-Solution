import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import datetime

# Title
st.title("AI-Powered Hospital Staffing Optimization")

# Sidebar - Upload data with enhanced visibility
st.sidebar.header("📂 Upload Patient Admission Data (CSV)")

st.markdown("### **Step 1: Upload Your CSV File**")

guide_text = """**📌 CSV Format:**  
- Column 1: `ds` (Date) - YYYY-MM-DD format  
- Column 2: `y` (Number of patient admissions)"""
st.info(guide_text)

uploaded_file = st.file_uploader("Click to Upload CSV File", type=["csv"], key="file_uploader")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    df.columns = ['ds', 'y']  # Prophet requires 'ds' (date) and 'y' (value)
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Show the uploaded data
    st.subheader("📊 Uploaded Data Preview")
    st.write(df.head())
    
    # Train Prophet Model
    model = Prophet()
    model.fit(df)
    
    # Forecast future patient admissions
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Plot forecast
    st.subheader("📈 Patient Admission Forecast")
    st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))
    
    # Predict staffing needs (Basic Rule: 1 Doctor per 10 Patients, 2 Nurses per Doctor)
    forecast['doctors_required'] = np.ceil(forecast['yhat'] / 10)
    forecast['nurses_required'] = forecast['doctors_required'] * 2
    forecast['support_staff_required'] = np.ceil(forecast['yhat'] / 20)
    
    # Show staffing needs
    st.subheader("👩‍⚕️ Predicted Staff Requirements")
    st.write(forecast[['ds', 'yhat', 'doctors_required', 'nurses_required', 'support_staff_required']].tail(10))
    
    # Plot staffing needs
    st.subheader("📌 Staff Allocation Plan")
    st.bar_chart(forecast[['ds', 'doctors_required', 'nurses_required', 'support_staff_required']].set_index('ds'))
    
    # Shift Scheduling (Basic Rotation)
    st.subheader("⏳ Shift Scheduling Recommendations")
    shifts = []
    for index, row in forecast.tail(7).iterrows():  # Generate shifts for last 7 days in forecast
        day = row['ds'].strftime('%Y-%m-%d')
        shifts.append({
            "Date": day,
            "Morning Shift": f"{int(row['doctors_required']/2)} Doctors, {int(row['nurses_required']/2)} Nurses, {int(row['support_staff_required']/2)} Support Staff",
            "Night Shift": f"{int(row['doctors_required']/2)} Doctors, {int(row['nurses_required']/2)} Nurses, {int(row['support_staff_required']/2)} Support Staff"
        })
    
    shifts_df = pd.DataFrame(shifts)
    st.write(shifts_df)
    
else:
    st.warning("⚠ Please upload a CSV file to proceed with forecasting and staff scheduling.")

# Footer
st.sidebar.info("Developed using Streamlit & Prophet")