import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import plotly.express as px
import plotly.io as pio
import analysis_and_plots

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to read tables from a single sheet in Excel data
def get_tables_from_sheet2(file_path, sheet_name):
    # Read the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    
    # Logic to identify and separate tables
    tables = []
    current_table = []
    for index, row in df.iterrows():
        if row.isnull().all():
            if current_table:
                # Convert to DataFrame and drop the first row
                table_df = pd.DataFrame(current_table).drop([0, 2, 9], errors='ignore')
                tables.append(table_df)
                current_table = []
        else:
            current_table.append(row)
    if current_table:
        # Convert to DataFrame and drop the first row
        table_df = pd.DataFrame(current_table).drop([0, 2, 9], errors='ignore')
        tables.append(table_df)
    
    return tables

@app.route('/')
def index():
    # Check if an uploaded file exists
    uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.xlsx')
    if not os.path.exists(uploaded_file_path):
        return render_template('index.html', tables="")

    # Load data from the uploaded Excel file
    sheet_name = "CFF"  # Replace with your sheet name
    tables = analysis_and_plots.get_tables_from_sheet(uploaded_file_path, sheet_name)
    # Process each table
    tables_html = ""
    '''''
    for i, df in enumerate(tables):
        # Replace NaN values with empty strings
        df = df.fillna('')

        if i == 0:
            tables_html += f"<h2>REVENUS</h2>"
        elif i == 1:
            tables_html += f"<h2>DEPENSES</h2>"
        elif i == 2:
            tables_html += f"<h2>Forcast</h2>"
        # Append the data table to the HTML
        tables_html += f"<div class='table-responsive'>{df.to_html(classes='table table-striped', index=False)}</div>"
    '''''
    # TABLE OF REVENUES
    revenus = tables[0]
    index_depenses = [
    "Loyer", "Restaurant", "Telephone", "Shopping", "Coffee",
    "Transports", "Electricte", "Netflix", "Divers Amazon", "Salle de sport",
    "Divers", "Autres"
    ]

    # TABLE OF REVENUES
    depenses = tables[2]
    index_revenus = ["Salaire"]

    # preprocessing TABLE OF REVENUES
    df_revenus = analysis_and_plots.preprocessing(revenus)


    # preprocessing TABLE OF DEPENSES
    df_depense = analysis_and_plots.preprocessing(depenses)
    print(df_depense)
    
    # Create dataframes for revenues and expenses
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

    print(forecast_balance)

    
    # VISUALIZATION OF THE DATAFRAME OF REVENUES
    tables_html += f"<h2>REVENUS</h2>"
    tables_html += f"<div class='table-responsive'>{dataframe_revenus.to_html(classes='table table-striped', index=False)}</div>"
    
    # VISUALIZATION OF THE DATAFRAME OF EXPENSES
    tables_html += f"<h2>DEPENSES</h2>"
    tables_html += f"<div class='table-responsive'>{dataframe_depenses.to_html(classes='table table-striped', index=False)}</div>"
    
    # VISUALIZATION OF THE DATAFRAME OF Balance
    tables_html += f"<h2>Balance</h2>"
    tables_html += f"<div class='table-responsive'>{df_main.transpose().to_html(classes='table table-striped', index=False)}</div>"
    
    # VISUALIZATION OF THE DATAFRAME OF Forecast
    tables_html += f"<h2>Forecast</h2>"
    tables_html += f"<div class='table-responsive'>{df_forecast.transpose().to_html(classes='table table-striped', index=False)}</div>"
    
    
    # Render the data tables and pie charts
    return render_template('index.html', tables=tables_html) # type: ignore

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.xlsx')
        file.save(file_path)
        return redirect(url_for('index'))
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)