import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import pickle
from datetime import datetime

def preprocessing(df):
    """
    Perform data preprocessing tasks on a DataFrame with columns 'Week' and 'Balance'.

    Parameters:
    - df (DataFrame): Input DataFrame with columns 'Week' and 'Balance'.

    Returns:
    - DataFrame: Preprocessed DataFrame with the same structure.

    Example:
    preprocessed_df = preprocessing(input_df)
    """


    df_1 = df.copy()
    df_1 = df_1.loc[:, df_1.columns.get_level_values(0) != 'Date']

    
    df_1 = df_1.dropna(how='all')

    #df_1.fillna("", inplace=True)
    df_1 = df_1.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    df_1.columns = [col.date() if isinstance(col, datetime) else col for col in df_1.columns]

    df_1.fillna("", inplace=True)
    df_1.columns = [col.date() if isinstance(col, datetime) else col for col in df_1.columns]

    return df_1





def train_test_split(df, split_index=455):
    """
    Splits a DataFrame into training and test sets.

    Parameters:
        df (DataFrame): The input DataFrame to be split.
        split_index (int): The index at which to split the DataFrame. Default is 455.

    Returns:
        train_data (DataFrame): The training data containing rows up to index (split_index-1).
        test_data (DataFrame): The test data containing rows from index split_index onwards.

    Example:
    train_data, test_data = train_test_split(df, split_index=400)
    """
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid value for 'split_index'. It should be within the range of the DataFrame.")

    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    return train_data, test_data





def exp_smoothing_model(train_data, test_data, model='triple', span=12):
    """
    Perform exponential smoothing to forecast balance.

    Parameters:
        train_data (DataFrame): Training data with 'Balance'.
        test_data (DataFrame): Test data with 'Week'.
        model (str): Smoothing model ('single', 'double', or 'triple'). Default is 'triple'.
        span (int): Span for single exponential smoothing. Default is 12.

    Returns:
        DataFrame: Forecasted balance with 'Week'.

    Example:
    predictions = exp_smoothing_model(train_data, test_data, model='triple')
    """
    time_steps = len(test_data)

    if model == 'triple':
        model_fit = ExponentialSmoothing(train_data['Balance'], trend='add', seasonal='add', seasonal_periods=span).fit()

    #Save the trained model to a file
    save_model(model_fit, filename='trained_exp_smoothing_model.pkl')
    
    test_predictions = model_fit.forecast(time_steps).rename('Balance')
    predictions = pd.DataFrame({'Week': test_data['Week'], 'Balance': test_predictions})

    return predictions


def rms_error_calc(test_data, predictions):
    """
    Calculate the RMSE between test and predicted data.

    Parameters:
    - test_data (DataFrame): Test data with 'Balance'.
    - predictions (DataFrame): Predictions with 'Balance'.

    Returns:
    - float: RMSE value.

    Example:
    rmse = rms_error_calc(test_df, predictions_df)
    """
    rms_error = np.sqrt(mean_squared_error(test_data['Balance'], predictions['Balance']))
    return rms_error




def save_model(trained_model, filename='trained_model.pkl'):
    """
    Save a trained model to a file.

    Parameters:
        trained_model: The trained model object.
        filename (str): The name of the file to save the model.
    """
    with open(filename, 'wb') as file:
        pickle.dump(trained_model, file)


def load_model(filename='trained_model.pkl'):
    """
    Load a trained model from a file.

    Parameters:
        filename (str): The name of the file containing the saved model.

    Returns:
        The loaded model object.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)



def predict_with_model(model, new_data, weeks_ahead=4):
    """
    Utilise un modèle pré-entraîné pour prédire les soldes sur de nouvelles données.

    Parameters:
        model: Le modèle pré-entraîné chargé.
        new_data (DataFrame): Les nouvelles données (avec la colonne 'Week').
        weeks_ahead (int): Nombre de semaines à prédire dans le futur.

    Returns:
        DataFrame: Un DataFrame contenant les prévisions pour les prochaines semaines.
    """
    # Assurez-vous que 'Week' est ordonné
    new_data = new_data.sort_values('Week')

    # Prédire pour les semaines futures
    last_date = pd.to_datetime(new_data['Week'].iloc[-1])
    future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, weeks_ahead + 1)]
    
    predictions = model.forecast(weeks_ahead)
    
    prediction_df = pd.DataFrame({
        'Week': future_dates,
        'Predicted_Balance': predictions
    })
    return prediction_df


def cumulative_addition(input_list, initial_value):
    """
    Takes an input list and an initial value. Produces an output list by 
    sequentially adding the cumulative sum to each element of the input list.

    Parameters:
        input_list (list): The list of numbers to process.
        initial_value (float or int): The initial value to start the cumulative sum.

    Returns:
        list: A new list with cumulative sums added.
    """
    output_list = []
    current_value = initial_value
    
    for value in input_list:
        current_value += value  # Add current value to the input element
        output_list.append(current_value)  # Append the result to the output list

    return output_list



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


def create_df(dframe, index_name, index):
    
    
    # Create the DataFrame
    df = pd.DataFrame(dframe)
    
    # Set the index and rename it
    df.index = index
    df.index.name = index_name
    
    return df



def sum_columns(dataframe):

    working_df = dataframe
    # Replace empty strings with 0
    #working_df.replace("", 0, inplace=True)
    working_df = working_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    # Calculate the sum for each column
    total_sum = working_df.sum(axis=0)

    # Convert the series to a DataFrame
    sum_df = total_sum.reset_index()

    # Rename the columns to 'Date' and 'Balance'
    sum_df.columns = ['Week', 'Balance']

    # Convert the 'Date' column to datetime format
    sum_df['Week'] = pd.to_datetime(sum_df['Week'])

    # Convert the 'Date' column to datetime format and format it as yyyy-mm-dd
    sum_df['Week'] = pd.to_datetime(sum_df['Week']).dt.strftime('%Y-%m-%d')

    return sum_df




def subtract_dataframes(df1, df2):
    """
    Subtracts the values of two DataFrames element-wise.
    
    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    
    Returns:
    pd.DataFrame: A new DataFrame containing the result of df1 - df2.
    """
    # Ensure that both DataFrames have the same shape and structure
    if df1.shape != df2.shape:
        raise ValueError("The DataFrames must have the same shape for subtraction.")
    
    # Perform subtraction (note that we subtract only the 'Balance' column)
    result_df = df1.copy()  # Make a copy of the first DataFrame
    result_df['Balance'] = df1['Balance'] - df2['Balance']  # Subtract the 'Balance' columns
    return result_df

def weeklySoldeForecast(actualSold,weeklyTotalCashFlow):
  return actualSold+weeklyTotalCashFlow


def weeks_between_dates(given_date):
    """
    Calcule le nombre de semaines entre la date actuelle et une date donnée.

    Parameters:
    - given_date (str): La date donnée au format "YYYY-MM-DD".

    Returns:
    - int: Le nombre de semaines (positif si dans le futur, négatif si dans le passé),
           avec un minimum de 1 si la différence est inférieure à 1 semaine.
    """
    current_date = datetime.now()
    delta = given_date - current_date
    
    return delta.days // 7


def Simulations(data, initial_balance: float, model_filename,df_main):
    """
    Calculate the forecasted balance based on input data.

    Args:
        data (List[Dict]): List of dictionaries containing 'date' and 'amount'.
        initial_balance (float): The initial balance to start the forecast.
        model_filename (str): Path to the pre-trained model file.

    Returns:
        List[float]: Forecasted balance including added amounts.
    """
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x['Date'])

    # Extract the maximum date
    max_date = max([datetime.strptime(entry['Date'], "%Y-%m-%d") for entry in sorted_data])

   
    weeks = int(weeks_between_dates(max_date))

    # Load the pre-trained model
    loaded_model = load_model(filename=model_filename)

    # Perform predictions
    predictions = predict_with_model(loaded_model, df_main, weeks_ahead=weeks)

    # Extract the predicted balances
    flux_forecast = predictions['Predicted_Balance']

    # Perform cumulative addition with the initial balance
    forecast_balance = cumulative_addition(flux_forecast, initial_balance)

    # Add the specified amounts from the data to the forecast balance
    if isinstance(forecast_balance, list):
        for entry in sorted_data:
            forecast_balance = [balance + entry['Amount'] for balance in forecast_balance]
        
        forecast_balance = ["{:.2f}".format(num) for num in forecast_balance]

    else:
        raise TypeError("forecast_balance must be a list. Check cumulative_addition implementation.")

    return forecast_balance



def update_spending_df(df, category, amount, spending_week):
    """
    Updates the DataFrame with a new spending entry.
    
    Parameters:
    df (pd.DataFrame): The current DataFrame of spending.
    category (str): The spending category.
    amount (float): The amount spent.
    spending_week (str): The week of the spending in 'YYYY-mm-dd' format.
    
    Returns:
    pd.DataFrame: The updated DataFrame.
    """
    # Ensure the spending_week is in 'YYYY-mm-dd' format and treat it as a Period object
    spending_week = pd.to_datetime(spending_week, format='%Y-%m-%d').to_period('W-SUN')  # Week ending on Sunday

    # Convert existing columns to Periods for consistency
    if not df.empty:
        existing_weeks = pd.to_datetime(df.columns, format='%Y-%m-%d', errors='coerce').to_period('W-SUN')
    else:
        existing_weeks = pd.PeriodIndex([], freq='W-SUN')
    
    # Handle missing weeks between the latest week and the new spending week
    if existing_weeks.size > 0:
        last_week = existing_weeks[-1]
    else:
        last_week = spending_week - 1  # Start from one week before the first spending week
    
    # Add missing weeks as columns
    all_weeks = pd.period_range(last_week + 1, spending_week, freq='W-SUN')
    for week in all_weeks:
        week_str = week.end_time.strftime('%Y-%m-%d')  # Convert the end of the week to 'YYYY-mm-dd'
        if week_str not in df.columns:
            df[week_str] = 0  # Add missing weeks with zero values
    
    # Ensure the spending week column exists
    spending_week_str = spending_week.end_time.strftime('%Y-%m-%d')
    if spending_week_str not in df.columns:
        df[spending_week_str] = 0
    
    # Ensure the category exists as a row
    if category not in df.index:
        df.loc[category] = [0] * len(df.columns)
    
    # Add the spending amount to the appropriate cell
    df.loc[category, spending_week_str] += amount
    
    # Sort the columns chronologically
    df = df.reindex(sorted(df.columns, key=lambda x: pd.to_datetime(x, format='%Y-%m-%d')), axis=1)
    
    return df


def plot_expense_pie_avg_percentage(df):
    """
    This function plots the average percentage of each expense category in the 'DEPENSES' row as a pie chart.
    
    Parameters:
    - df: The DataFrame containing the expense data.
    """
    # Calculate the average for each category across all time periods
    category_averages = df.mean(axis=1)  # Average over the rows (axis=1)
    
    # Calculate the total of all categories
    total_avg = category_averages.sum()
    
    # Calculate the percentage for each category
    category_percentages = (category_averages / total_avg) * 100
    
    # Plot the results as a pie chart
    plt.figure(figsize=(8, 8))
    category_percentages.plot(kind='pie', labels=category_percentages.index, autopct='%1.1f%%', startangle=90, cmap="Set3")

    # Add a title
    plt.title('Average Percentage of Each Category of DEPENSES')

    # Show the plot
    plt.ylabel('')  # Remove the default y-axis label
    plt.show()
