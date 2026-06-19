import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Adib M&V Web Tool", layout="wide")

# --- HARDCODED WEATHER DATABASE (CDD) ---
CDD_DB = {
    "2019": [132.0, 132.0, 160.0, 144.0, 155.0, 135.0, 140.0, 138.0, 139.0, 96.0, 105.0, 107.0],
    "2023": [122.4, 143.1, 158.2, 131.7, 162.8, 151.2, 136.8, 128.9, 132.7, 121.6, 115.8, 108.8],
    "2024": [122.4, 143.1, 158.2, 150.8, 162.6, 135.1, 169.2, 131.6, 135.8, 137.4, 110.2, 113.7],
    "2025": [108.9, 117.9, 137.9, 131.9, 165.0, 164.8, 175.7, 144.6, 129.1, 133.4, 119.4, 105.8]
}

# --- SESSION & SERVER MEMORY SETUP ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None

if 'model_data' not in st.session_state or st.session_state['model_data'] is None:
    if os.path.exists("saved_baseline.pkl"):
        with open("saved_baseline.pkl", "rb") as f:
            st.session_state['model_data'] = pickle.load(f)
    else:
        st.session_state['model_data'] = None

# --- CALLBACK FUNCTIONS ---
def trigger_logout():
    st.session_state['logged_in'] = False
    st.session_state['role'] = None

# --- MAIN APPLICATION LOGIC ---
if not st.session_state['logged_in']:
    # ==========================================
    # LOGIN SCREEN
    # ==========================================
    st.title("🔐 Login to M&V Web Portal")
    st.write("Please enter your credentials to access the tool.")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login", type="primary"):
        if username == "adib" and password == "admin123":
            st.session_state['logged_in'] = True
            st.session_state['role'] = "Admin"
            st.rerun()
        elif username == "staff" and password == "user123":
            st.session_state['logged_in'] = True
            st.session_state['role'] = "User"
            st.rerun()
        else:
            st.error("Incorrect username or password")

else:
    # ==========================================
    # AUTHENTICATED VIEW (Logged In)
    # ==========================================
    st.sidebar.title(f"Welcome, {st.session_state['role']}")
    st.sidebar.button("Logout", on_click=trigger_logout)
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by Adib")
    
    st.title("⚡ Energy Baseline M&V Tool")

    # ------------------------------------------
    # ADMIN ONLY VIEW: Step 1 Baseline Setup
    # ------------------------------------------
    if st.session_state['role'] == "Admin":
        st.header("Step 1: Admin Baseline Setup")
        
        base_year = st.selectbox("Select Baseline Year:", ["2019", "2023", "2024", "2025"], key="base_yr")
        base_file = st.file_uploader("Upload Baseline CSV (12 Months)", type=['csv'], key="base_csv")
        
        if base_file:
            df_baseline = pd.read_csv(base_file)
            
            # Auto-Inject CDD Data
            if len(df_baseline) == 12:
                df_baseline['CDD'] = CDD_DB[base_year]
                st.write("Preview (with auto-injected CDD):", df_baseline.head(3))
                
                cols = list(df_baseline.columns)
                target_y = st.selectbox("Select Energy (Y):", cols)
                selected_x_vars = st.multiselect("Select Variables (X):", cols, default=["CDD"])
                
                if st.button("Run MLR Baseline Analysis", type="primary"):
                    if target_y and selected_x_vars:
                        clean_df = df_baseline[selected_x_vars + [target_y]].dropna()
                        X = clean_df[selected_x_vars]
                        y = clean_df[target_y]
                        
                        # Train Model
                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        
                        # --- NEW COMPLIANCE MATH INCORPORATED HERE ---
                        n = len(X)
                        p = len(selected_x_vars)
                        y_mean = np.mean(y)
                        
                        r2 = r2_score(y, y_pred)
                        rmse = np.sqrt(np.sum((y - y_pred)**2) / (n - p - 1))
                        cv_rmse = (rmse / y_mean) * 100
                        nmbe = (np.sum(y - y_pred) / ((n - p - 1) * y_mean)) * 100
                        
                        # Save model & metrics to server
                        st.session_state['model_data'] = {
                            'model': model, 'vars': selected_x_vars, 'r2': r2, 'cv_rmse': cv_rmse, 'nmbe': nmbe
                        }
                        
                        with open("saved_baseline.pkl", "wb") as f:
                            pickle.dump(st.session_state['model_data'], f)
                            
                        # Display Compliance Results
                        st.success("✅ Model Trained & Saved to Server!")
                        st.write("### Baseline Compliance Metrics")
                        st.write(f"**R² Score:** {r2:.4f} *(Target: > 0.75)*")
                        st.write(f"**CV(RMSE):** {cv_rmse:.2f}% *(Target: < 15%)*")
                        st.write(f"**NMBE:** {nmbe:.2f}% *(Target: < ±5%)*")
                    else:
                        st.warning("Please select Y and at least one X variable.")
            else:
                st.error(f"⚠️ Error: Your CSV has {len(df_baseline)} rows. It must have exactly 12 rows (Jan-Dec).")
        st.markdown("---")

    # ------------------------------------------
    # ALL USERS VIEW: Step 2 Reporting Period
    # ------------------------------------------
    st.header("Step 2: Reporting Period Analysis")
    
    if st.session_state['model_data'] is None:
        st.info("⚠️ Waiting for Admin to setup the Baseline Model.")
    else:
        st.success("✅ Baseline Model is Active and Ready.")
        
        rep_year = st.selectbox("Select Reporting Year:", ["2019", "2023", "2024", "2025"], key="rep_yr")
        rep_file = st.file_uploader("Upload Reporting CSV (12 Months)", type=['csv'], key="rep_csv")
        
        if rep_file:
            df_reporting = pd.read_csv(rep_file)
            
            # Auto-Inject CDD Data
            if len(df_reporting) == 12:
                df_reporting['CDD'] = CDD_DB[rep_year]
                st.write("Preview (with auto-injected CDD):", df_reporting.head(3))
                
                y_col = st.selectbox("Select Actual Energy (Y):", list(df_reporting.columns))
                
                if st.button("Calculate Energy Savings", type="primary"):
                    model_data = st.session_state['model_data']
                    x_vars = model_data['vars']
                    
                    missing = [col for col in x_vars if col not in df_reporting.columns]
                    if missing:
                        st.error(f"Missing required columns in CSV: {missing}.")
                    else:
                        clean_rep = df_reporting[x_vars + [y_col]].dropna()
                        X_rep = clean_rep[x_vars]
                        y_actual = clean_rep[y_col]
                        
                        adjusted_baseline = model_data['model'].predict(X_rep)
                        total_savings = (adjusted_baseline - y_actual).sum()
                        
                        st.subheader(f"💰 Total Savings: {total_savings:,.2f} kWh")
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(clean_rep.index, adjusted_baseline, label='Adjusted Baseline', color='orange', linestyle='--')
                        ax.plot(clean_rep.index, y_actual, label='Actual Energy Used', color='blue')
                        ax.fill_between(clean_rep.index, adjusted_baseline, y_actual, where=(adjusted_baseline > y_actual), color='green', alpha=0.15, label='Savings')
                        ax.fill_between(clean_rep.index, adjusted_baseline, y_actual, where=(adjusted_baseline <= y_actual), color='red', alpha=0.15, label='Overconsumption')
                        
                        ax.set_title('IPMVP Option C (MLR): Reporting Period Savings')
                        ax.set_ylabel('Energy (kWh)')
                        ax.legend()
                        ax.grid(True, alpha=0.5)
                        
                        st.pyplot(fig)
            else:
                st.error(f"⚠️ Error: Your CSV has {len(df_reporting)} rows. It must have exactly 12 rows (Jan-Dec).")
