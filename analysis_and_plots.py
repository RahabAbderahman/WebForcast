import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import pickle

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

    df_1.fillna("", inplace=True)
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

def plot_train_test_data(train_data, test_data, x_label_index=50, figsize=(12, 5), dpi=150):
    """
    Create a line plot to visualize training and test data.

    Parameters:
    - train_data (DataFrame): DataFrame containing training data with columns 'Week' and 'Balance'.
    - test_data (DataFrame): DataFrame containing test data with columns 'Week' and 'Balance'.
    - x_label_index (int): Frequency for displaying x-axis labels.
    - figsize (tuple): Size of the figure.
    - dpi (int): Dots per inch for the figure.

    Returns:
    - None: Displays the plot.

    Example:
    plot_train_test_data(train_df, test_df)
    """
    combined_data = pd.concat([train_data, test_data])
    xlabels = combined_data['Week'][::x_label_index]

    sns.set(style="whitegrid")
    plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(data=train_data, x='Week', y='Balance', label='TRAIN')
    sns.lineplot(data=test_data, x='Week', y='Balance', label='TEST')

    plt.xticks(xlabels)
    plt.xlabel('Week')
    plt.ylabel('Balance')
    plt.title('Balance Data')
    plt.legend()
    plt.tight_layout()
    plt.show()

def seasonal_decompose_plot(train_data, period=12):
    """
    Perform seasonal decomposition on training data and plot the components.

    Parameters:
        train_data (DataFrame): Training data with 'Week' and 'Balance'.
        period (int): Seasonal period. Default is 7.

    Returns:
        None

    Example:
    seasonal_decompose_plot(train_data, period=12)
    """
    result = seasonal_decompose(train_data['Balance'], period=period, model='additive')

    plt.figure(figsize=(8, 6))
    plt.subplot(411)
    plt.plot(result.observed, label='Observed')
    plt.legend()
    plt.subplot(412)
    plt.plot(result.trend, label='Trend')
    plt.legend()
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend()
    plt.subplot(414)
    plt.plot(result.resid, label='Residual')
    plt.legend()

    plt.suptitle('Seasonal Decomposition of Balance Data')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

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

    if model == 'single':
        alpha = 2 / (span + 1)
        model_fit = SimpleExpSmoothing(train_data['Balance']).fit(smoothing_level=alpha)
    elif model == 'double':
        model_fit = ExponentialSmoothing(train_data['Balance'], trend='add').fit()
    elif model == 'triple':
        model_fit = ExponentialSmoothing(train_data['Balance'], trend='add', seasonal='add', seasonal_periods=span).fit()

    #Save the trained model to a file
    save_model(model_fit, filename='trained_exp_smoothing_model.pkl')
    
    test_predictions = model_fit.forecast(time_steps).rename('Balance')
    predictions = pd.DataFrame({'Week': test_data['Week'], 'Balance': test_predictions})

    return predictions

def exponential_smoothing_plot(train_data, test_data, predictions, split_index=100, figsize=(12, 6), dpi=150):
    """
    Create a line plot to visualize training, test, and predicted data.

    Parameters:
    - train_data (DataFrame): Training data with 'Week' and 'Balance'.
    - test_data (DataFrame): Test data with 'Week' and 'Balance'.
    - predictions (DataFrame): Predictions with 'Week' and 'Balance'.
    - split_index (int): Frequency for x-axis labels.
    - figsize (tuple): Figure size.
    - dpi (int): Figure resolution.

    Returns:
    - None: Displays the plot.

    Example:
    exponential_smoothing_plot(train_df, test_df, predictions)
    """
    combined_data = pd.concat([train_data, test_data])
    xlabels = combined_data['Week'][::split_index]

    sns.set(style="whitegrid")
    plt.figure(figsize=figsize, dpi=dpi)
    sns.lineplot(data=train_data, x='Week', y='Balance', label='TRAIN')
    sns.lineplot(data=predictions, x='Week', y='Balance', label='PREDICTION')
    sns.lineplot(data=test_data, x='Week', y='Balance', label='TEST')

    plt.xticks(xlabels)
    plt.xlabel('Week')
    plt.ylabel('Balance')
    plt.title('Balance Data')
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    new_data = new_data.sort_values(by='Week')

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
    working_df.replace("", 0, inplace=True)
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