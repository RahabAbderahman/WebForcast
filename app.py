import pandas as pd
from flask import Flask, render_template
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Function to read tables from a single sheet in Excel data
def get_tables_from_sheet(file_path, sheet_name):
    # Read the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    
    # Logic to identify and separate tables
    tables = []
    current_table = []
    for index, row in df.iterrows():
        if row.isnull().all():
            if current_table:              
                table_df = pd.DataFrame(current_table)
                if index <= 1:
                    table_df=table_df.drop(1)                   
                tables.append(table_df)
                current_table = []
        else:
            current_table.append(row)
    if current_table:
        tables.append(pd.DataFrame(current_table))
    
    return tables

@app.route('/')
def index():
    # Load data from Excel file
    file_path = "data.xlsx"  # Replace with your Excel file
    sheet_name = "CFF"  # Replace with your sheet name
    tables = get_tables_from_sheet(file_path, sheet_name)

    # Process each table
    tables_html = ""
    pie_charts_html = ""
    table1, table2, table3 = [], [], []
    for i, df in enumerate(tables):
        # Replace NaN values with empty strings
        df = df.fillna('')
        
        if i == 0:
            table1 = df  # Table des REVENUS
            tables_html += f"<h2>REVENUS</h2>"
        elif i == 1:
            table2 = df  # Table des DEPENSES
            tables_html += f"<h2>DEPENSES</h2>"

        elif i == 2:
            table3 = df  # Table de Forcast
            tables_html += f"<h2>Forcast</h2>"
        
        # Append the data table and pie chart to the HTML
        tables_html += df.to_html(classes='table table-striped', index=False)
    

    # Render the data tables and pie charts$
    return render_template('index.html', tables=tables_html)

if __name__ == '__main__':
    app.run(debug=True)