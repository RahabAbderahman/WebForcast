from datetime import datetime
import os
import analysis_and_plots
import pandas as pd
import streamlit as st
from io import BytesIO
import plotly.express as px

# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Define the uploaded_file variable at the top level
#uploaded_file = None

st.title('Forecast Data Visualization')


#General variables
#df_forecast= None
#dataframe_depenses = None
#dataframe_revenus = None
if "df_main" not in st.session_state:
        st.session_state.df_main = None
if "dataframe_depenses" not in st.session_state:
        st.session_state.dataframe_depenses = None
if "dataframe_revenus" not in st.session_state:
        st.session_state.dataframe_revenus = None
if "result" not in st.session_state:
        st.session_state.result = None
if "solde_before" not in st.session_state:
        st.session_state.solde_before = None
if "df_simulation" not in st.session_state:
        st.session_state.df_simulation = None
if 'purchases_table' not in st.session_state:
        st.session_state.purchases_table = []
if 'purchases_table2' not in st.session_state:
        st.session_state.purchases_table2 = []

if 'tables' not in st.session_state:
        st.session_state.tables = None
# Create two columns
col1, col2 = st.columns(2)

with col2:

    # Initialize the uploaded_file variable using the file uploader
     
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    st.session_state.uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    #st.session_state.uploaded_file = uploaded_file

    if 'actual_balance' not in st.session_state:
        st.session_state.actual_balance = 0.0


    st.session_state.ancien_balance = st.session_state.actual_balance
    # Add input text and button for actual balance
    st.session_state.actual_balance=st.number_input("Enter Actual Balance", format="%.2f")



    #***************** Validate Button **************************

    # Initialize session state for validation
    if 'validate' not in st.session_state:
        st.session_state.validate = False

    # Enable the validate button only if a file is uploaded and actual balance is provided
    if st.session_state.uploaded_file is not None and st.session_state.actual_balance != 0.0:
        if st.button("Validate", key="ValidateBtn", disabled=False):
            st.session_state.validate = True
            sheet_name = "CFF"  # Replace with your sheet name
            st.session_state.tables = analysis_and_plots.get_tables_from_sheet(st.session_state.uploaded_file, sheet_name)
            # Example of processing and visualizing data
            
            # TABLE OF REVENUES
            revenus = st.session_state.tables[0]
            index_revenus = ["Salaire"]
            print('revenus',revenus)
            # TABLE OF Expenses
            depenses = st.session_state.tables[2]
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
            st.session_state.dataframe_revenus = analysis_and_plots.create_df(df_revenus,"REVENUS",index_revenus)

            # Create dataframes for expenses
            st.session_state.dataframe_depenses = analysis_and_plots.create_df(df_depense,"DEPENSES",index_depenses)
            total_depenses =  analysis_and_plots.sum_columns(st.session_state.dataframe_depenses)
            
            total_revenus = analysis_and_plots.sum_columns(st.session_state.dataframe_revenus)
            # Create a dataframe for the main table
            st.session_state.df_main = analysis_and_plots.subtract_dataframes(total_revenus,total_depenses)  
            # Charger le modèle pré-entraîné
            loaded_model = analysis_and_plots.load_model(filename='trained_exp_smoothing_model.pkl')

            # Prédire sur plusieurs semaines pour le nouveau client
            predictions = analysis_and_plots.predict_with_model(loaded_model, st.session_state.df_main, weeks_ahead=4)
            flux_forecast = []

            flux_forecast = predictions['Predicted_Balance']


            forecast_balance = analysis_and_plots.cumulative_addition(flux_forecast,st.session_state.actual_balance)
            st.session_state.solde_before = analysis_and_plots.cumulative_addition(st.session_state.df_main['Balance'],st.session_state.actual_balance)
            st.session_state.solde_before = ["{:.2f}".format(num) for num in st.session_state.solde_before]
            forecast_balance = ["{:.2f}".format(num) for num in forecast_balance]
            st.session_state.result = [st.session_state.solde_before + [''] * (len(forecast_balance) - 1), [''] * len(st.session_state.solde_before) + forecast_balance]
            index_Forcast = ["Actuel Balance","Forcast"]
            dates = pd.to_datetime(st.session_state.dataframe_revenus.columns.tolist())
            st.session_state.result = analysis_and_plots.create_df3(st.session_state.result,'FORECAST', index_Forcast, dates)
            df_forecast= pd.DataFrame(forecast_balance, columns=['Forecast_Balance'])
            
    else:
        st.button("Validate", disabled=True)

    # ******************  Purshace Table **************
    
    st.header("Purchases Table")

    def add_row():
        st.session_state.purchases_table.append({"Description": "", "Category": "Loyer", "Amount": "","Date": datetime.today().strftime("%Y-%m-%d"),"Delete":""})
        st.rerun()
        
        
    def delete_row(index):
        st.session_state.purchases_table.pop(index)
        st.rerun()        

    for i, row in enumerate(st.session_state.purchases_table):
        cols = st.columns(5)
        row["Description"] = cols[0].text_input("Description", value=row["Description"], key=f"description_{i}")
        # Category selectbox with "Autres" option
        categories = ["Loyer", "Restaurant", "Telephone", "Shopping", "Coffee", "Transports", "Electricte", "Netflix", "Divers Amazon", "Salle de sport", "Divers", "Autres"]
        row["Category"] = cols[1].selectbox("Category", categories, key=f"category_{i}")
        row["Amount"] = cols[2].number_input("Amount", format="%.2f", key=f"amount_{i}")
        row["Date"] = cols[3].date_input("Date", key=f"date_{i}")
        row["Delete"]= cols[4].button("Delete",  key=f"delete_{i}", on_click=delete_row, args=(i,))
            
    
    #****************** Add Purchase Button ***************************
    if st.button("Add Purchase"):
        add_row()
                    


with col1: 

    if st.session_state.dataframe_revenus is not None :
        st.header("REVENUS")
        st.dataframe(st.session_state.dataframe_revenus)
    if st.session_state.dataframe_depenses is not None:
        st.header("DEPENSES")
        st.dataframe(st.session_state.dataframe_depenses)
            
            

with col2:
    # Add a button to simulate the forecast
    if len( st.session_state.purchases_table)!=0 and st.session_state.actual_balance != 0.0:
        if st.button("Simulate") :
            st.session_state.df_simulation= analysis_and_plots.Simulations(st.session_state.purchases_table,st.session_state.actual_balance ,"trained_exp_smoothing_model.pkl",st.session_state.df_main)
            st.session_state.result = [st.session_state.solde_before + [''] * (len(st.session_state.df_simulation) - 1), [''] * len(st.session_state.solde_before) + st.session_state.df_simulation]
            
            
            for purchase in st.session_state.purchases_table:
                st.session_state.dataframe_depenses = analysis_and_plots.update_spending_df(st.session_state.dataframe_depenses,purchase["Category"],purchase["Amount"],purchase["Date"])
            
        
            index_simulation = ["Actuel  Balance", "SIMULATION"]
            dates = pd.to_datetime(st.session_state.dataframe_revenus.columns.tolist(), format= "%Y-%m-%d")
            st.session_state.result = analysis_and_plots.create_df2(st.session_state.result, 'Simulation', index_simulation, dates, st.session_state.purchases_table, st.session_state.purchases_table2)
            st.session_state.purchases_table2 =st.session_state.purchases_table
            #st.session_state.purchases_table= []
            st.rerun()
    else : st.button("Simulate", disabled=True)    
             
        
with col1:
    
    if st.session_state.result is not None:
        st.dataframe(st.session_state.result)
    
with col2: 

    # Section to download an Excel file
    st.header("Download Provided Excel File")

    # Load the file from the repository
    file_path = "CFF - guide.xlsx"
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Provide the download button
    st.download_button(
        label="Download CFF.xlsx",
        data=file_data,
        file_name="CFF - guide.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )