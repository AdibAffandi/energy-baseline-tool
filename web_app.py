import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Adib M&V Web Tool", layout="wide")

# --- SESSION STATE SETUP ---
if 'model_data' not in st.session_state:
    st.session_state['model_data'] = None

# We add a session state for the password so we can clear it later
if 'admin_pwd' not in st.session_state:
    st.session_state['admin_pwd'] = ""

# --- SIDEBAR: ADMIN ACCESS ---
st.sidebar.title("⚙️ Admin Access")

# Notice we linked this input box to the 'admin_pwd' key
st.sidebar.text_input("Admin Password", type="password", key="admin_pwd")

# The app checks if the memory vault contains the correct password
is_admin = (st.session_state['admin_pwd'] == "admin123")

if is_admin:
    st.sidebar.success("Admin Mode Unlocked")
    
    # ---> NEW FEATURE: LOGOUT BUTTON <---
    if st.sidebar.button("Logout"):
        st.session_state['admin_pwd'] = ""  # This deletes the password
        st.rerun()  # This instantly refreshes the page to hide Step 1
        
st.sidebar.markdown("---")
st.sidebar.caption("Adib Affandi")

st.title("⚡ Energy Baseline M&V Tool")

# ==========================================
# ADMIN ONLY VIEW: Step 1 Baseline Setup
# ==========================================
if is_admin:
    st.header("Step 1: Admin Baseline Setup")
    base_file = st.file_uploader("Upload Baseline CSV", type=['csv'], key="base")
    
    if base_file:
        df_baseline = pd.read_csv(base_file)
        st.write("Preview:", df_baseline.head(3))
        
        cols = list(df_baseline.columns)
        target_y = st.selectbox("Select Energy (Y):", cols)
        selected_x_vars = st.multiselect("Select Variables (X):", cols)
        
        if st.button("Run MLR Baseline Analysis", type="primary"):
            if target_y and selected_x_vars:
                clean_df = df_baseline[selected_x_vars + [target_y]].dropna()
                X = clean_df[selected_x_vars]
                y = clean_df[target_y]
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                n = len(X)
                p = len(selected_x_vars)
                se = np.sqrt(np.sum((y - y_pred)**2) / (n - p - 1))
                
                # Save model to session state
                st.session_state['model_data'] = {
                    'model': model, 'vars': selected_x_vars, 'se': se, 'r2': r2
                }
                st.success(f"✅ Model Trained Successfully! R²: {r2:.4f}")
            else:
                st.warning("Please select Y and at least one X variable.")
    st.markdown("---")

# ==========================================
# PUBLIC VIEW: Step 2 Reporting Period
# ==========================================
st.header("Step 2: Reporting Period Analysis")

if st.session_state['model_data'] is None:
    st.info("⚠️ Waiting for Admin to setup the Baseline Model. (Enter Admin Password in Sidebar)")
else:
    st.success("✅ Baseline Model is Active and Ready.")
    rep_file = st.file_uploader("Upload Reporting CSV", type=['csv'], key="rep")
    
    if rep_file:
        df_reporting = pd.read_csv(rep_file)
        
        # CSV Preview for the staff
        st.write("Preview:", df_reporting.head(3))
        
        y_col = st.selectbox("Select Actual Energy (Y):", list(df_reporting.columns))
        
        if st.button("Calculate Energy Savings", type="primary"):
            model_data = st.session_state['model_data']
            x_vars = model_data['vars']
            
            missing = [col for col in x_vars if col not in df_reporting.columns]
            if missing:
                st.error(f"Missing required columns in CSV: {missing}")
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