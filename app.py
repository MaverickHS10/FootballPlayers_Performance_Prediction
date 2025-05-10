import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = ['Min', 'xG', 'xAG', 'Sh', 'SoT', 'KP', 'PrgP', 'PrgC', 'SCA90', 'Age']

st.set_page_config(page_title="G+A Predictor", layout="wide")
st.title(" Player G+A Predictor")

tab1, tab2 = st.tabs([" Bulk Prediction (CSV)", "🧍 Single Player Prediction"])

#bulk prediction
with tab1:
    st.subheader("Upload CSV with Player Stats")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            missing = [col for col in feature_columns if col not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                X_new = df[feature_columns]
                X_scaled = scaler.transform(X_new)
                predictions = model.predict(X_scaled)

                df['Predicted_G+A'] = predictions
                st.success(" Predictions generated!")

                st.dataframe(df[['Player', 'Predicted_G+A']])

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                label=" Download Predictions as CSV",
                data=csv,
                file_name='predictedPlayers.csv',
                mime='text/csv'
                )

                st.write("### Bar Chart of Predicted G+A")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(df['Player'], df['Predicted_G+A'], color='steelblue')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Predicted G+A")
                st.pyplot(fig)

                N = st.slider("Show Top N Players", 3, 20, 5)
                df_top = df.sort_values(by='Predicted_G+A', ascending=False).head(N)

                st.write(f"### Top {N} Players by Predicted G+A")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.barh(df_top['Player'], df_top['Predicted_G+A'], color='lightgreen')
                ax2.invert_yaxis()
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error reading file: {e}")

#single player
with tab2:
    st.subheader("Enter Stats for a Single Player")

    with st.form("single_player_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Min = st.number_input("Minutes Played", value=1000)
            xG = st.number_input("Expected Goals (xG)", value=5.0)
            xAG = st.number_input("Expected Assists (xAG)", value=2.0)
            Sh = st.number_input("Shots", value=30)
        with col2:
            SoT = st.number_input("Shots on Target", value=15)
            KP = st.number_input("Key Passes", value=20)
            PrgP = st.number_input("Progressive Passes", value=40)
        with col3:
            PrgC = st.number_input("Progressive Carries", value=35)
            SCA90 = st.number_input("Shot-Creating Actions/90", value=3.5)
            Age = st.number_input("Age", value=24)

        submitted = st.form_submit_button("Predict G+A")

    if submitted:
        input_data = np.array([[Min, xG, xAG, Sh, SoT, KP, PrgP, PrgC, SCA90, Age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.success(f" Predicted G+A: **{prediction:.2f}**")
