from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Function to read Excel data
def get_data_from_excel(file_path):
    # Replace 'data.xlsx' with your actual Excel file path
    return pd.read_excel(file_path)

@app.route('/')
def index():
    # Load data from Excel file
    file_path = "data.xlsx"  # Replace with your Excel file
    df = get_data_from_excel(file_path)

    # Replace NaN values with empty strings
    df = df.fillna('')

    # Create a pie chart using Plotly
    pie_fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Data Distribution")
    pie_chart_html = pio.to_html(pie_fig, full_html=False)

    # Render the data table and pie chart
    return render_template('index.html', table=df.to_html(classes='table table-striped', index=False), pie_chart=pie_chart_html)

if __name__ == '__main__':
    app.run(debug=True)
