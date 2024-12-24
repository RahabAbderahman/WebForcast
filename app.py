import os
import analysis_and_plots
import pandas as pd
import streamlit as st

# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Define the uploaded_file variable at the top level
uploaded_file = None

st.title('Forecast Data Visualization')

# Create two columns
col1, col2 = st.columns(2)

with col2:
    st.header("Purchases Table")

    if 'purchases_table' not in st.session_state:
        st.session_state.purchases_table = []

    def add_row():
        st.session_state.purchases_table.append({"Description": "", "Category": "", "Amount": "", "Delete": ""})

    def delete_row(index):
        st.session_state.purchases_table.pop(index)

    

    for i, row in enumerate(st.session_state.purchases_table):
        cols = st.columns(4)
        row["Description"] = cols[0].text_input("Description", value=row["Description"], key=f"description_{i}")
        # Category selectbox with "Other" option
        categories = ["Loyer", "Restaurant", "Telephone", "Shopping", "Coffee", "Transports", "Electricte", "Netflix", "Divers Amazon", "Salle de sport", "Divers", "Autres"]
        selected_category = row.get("Category", "Loyer")
        if selected_category not in categories:
            selected_category = "Loyer"
        selected_category = cols[1].selectbox("Category", categories, index=categories.index(selected_category), key=f"category_{i}")
        
        if selected_category == "Autres":
            row["Category"] = cols[1].text_input("Specify Category", value=row.get("Category", ""), key=f"other_category_{i}")
        else:
            row["Category"] = selected_category
        
        row["Amount"] = cols[2].text_input("Amount", value=row["Amount"], key=f"amount_{i}")
        if cols[3].button("Delete", key=f"delete_{i}"):
            delete_row(i)

    if st.button("Add Purchase"):
        add_row()
    # Initialize the uploaded_file variable using the file uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    

if uploaded_file is not None:
    # Store the uploaded file in session state
    st.session_state.uploaded_file = uploaded_file

with col1:
    if 'uploaded_file' in st.session_state:
        sheet_name = "CFF"  # Replace with your sheet name
        tables = analysis_and_plots.get_tables_from_sheet(st.session_state.uploaded_file, sheet_name)
        # Example of processing and visualizing data
        
        # TABLE OF REVENUES
        revenus = tables[0]
        index_revenus = ["Salaire"]
        # TABLE OF Expenses
        depenses = tables[1]
        index_depenses = [
        "Loyer", "Restaurant", "Telephone", "Shopping", "Coffee",
        "Transports", "Electricte", "Netflix", "Divers Amazon", "Salle de sport",
        "Divers", "Autres"
        ]

        # Preprocessing TABLE OF REVENUES
        df_revenus = analysis_and_plots.preprocessing(revenus)

        # Preprocessing TABLE OF DEPENSES
        df_depense = analysis_and_plots.preprocessing(depenses)

        # Create dataframes for revenues
        dataframe_revenus = analysis_and_plots.create_df(df_revenus,"REVENUS",index_revenus)
    
        # Create dataframes for expenses
        dataframe_depenses = analysis_and_plots.create_df(df_depense,"DEPENSES",index_depenses)


        total_depenses =  analysis_and_plots.sum_columns(dataframe_depenses)
        total_revenus = analysis_and_plots.sum_columns(dataframe_revenus)
        
        # Create a dataframe for the main table
        df_main = analysis_and_plots.subtract_dataframes(total_revenus,total_depenses)

        # Charger le modèle pré-entraîné
        loaded_model = analysis_and_plots.load_model(filename='trained_exp_smoothing_model.pkl')


        # Prédire sur plusieurs semaines pour le nouveau client
        predictions = analysis_and_plots.predict_with_model(loaded_model, df_main, weeks_ahead=4)

        flux_forecast = []

        flux_forecast = predictions['Predicted_Balance']


        forecast_balance = analysis_and_plots.cumulative_addition(flux_forecast,1564.01)
        df_forecast= pd.DataFrame(forecast_balance, columns=['Forecast_Balance'])
        
        st.header("REVENUS")
        st.dataframe(dataframe_revenus)
        st.header("DEPENSES")
        st.dataframe(dataframe_depenses)
        st.header("Balance")
        st.dataframe(df_main.transpose())
        st.header("FORECAST")        
        st.dataframe(df_forecast.transpose())
        