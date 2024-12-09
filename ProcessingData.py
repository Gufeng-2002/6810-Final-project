
import kagglehub
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import \
StandardScaler, OneHotEncoder, FunctionTransformer, PowerTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


# read data from local
def read_data(type = 'sample'):
    """Read data set into a dataframe

    Args:
        type (str, optional): _description_. Defaults to 'sample', reading the sample data(2000,27),
        if 'complete', then read the whole (20000, 27) data set
    """
    data_path = "/Users/gufeng/2024_Fall/dasc6810/6810 Final project/data"
    if type == 'sample':
        data = pd.read_csv(data_path + "/sampled_df_trian.csv", index_col=0)
    elif type == 'complete':
        data = pd.read_csv(data_path + "/raw_data.csv").drop("Unnamed: 0", axis = 1)
    return data     

def check_dataframe_info(df):
    """
    Quickly check the data types and whether there are missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        pd.DataFrame: A summary of column data types, missing values, and missing value percentages.
    """
    # Get data types
    data_types = df.dtypes

    # Count missing values
    missing_values = df.isnull().sum()

    # Calculate percentage of missing values
    missing_percentage = (missing_values / len(df)) * 100

    # Create a summary DataFrame
    summary = pd.DataFrame({
        "Data Type": data_types,
        "Missing Values": missing_values,
        "Missing Percentage (%)": missing_percentage
    }).sort_values(by="Missing Values", ascending=False)

    return summary
    
# plots function
def _plot_numerical_distribution(dataframe, plot_type='hist', des_string=False, y=None):
    """
    Quickly draw plots for all columns in a dataframe.
    
    Args:
        dataframe (pd.DataFrame): A pandas dataframe where all columns are numerical.
        plot_type (str): The type of plots to draw. Options: 'hist', 'reg', 'box'.
        des_string (bool): Whether or not to add sample statistics to the plots.
        y (str): If plot_type is 'reg', the corresponding dependent variable.
    """
    # Only apply on the numerical columns
    dataframe = dataframe.select_dtypes(include=['number'])

    # Descriptive stats function
    def stats_descriptive_string(series):
        """Receive a series and return a string of common statistical features."""
        des_series = series.describe()[1:3].round().astype("str")
        des_string = ''
        for feature, value in zip(des_series.index, des_series):
            des_string += feature
            des_string += ": "
            des_string += value
            des_string += "\n"
        return des_string

    # Add descriptive text to the plot
    def add_text(ax, text):
        ax.text(
            0.95, 0.95,  # X and Y positions (normalized to the axes, 1 is the top/right)
            text,  # The text to display
            transform=ax.transAxes,  # Use axes coordinates
            ha="right",  # Horizontal alignment
            va="top",  # Vertical alignment
            fontsize=10,  # Text size
            color="red"  # Text color
        )

    # Plot based on the type
    def _plot(plot_type=plot_type):
        if plot_type == "hist":
            sns.histplot(dataframe[var], kde=True, ax=ax)
        elif plot_type == 'reg' and y is not None:
            sns.regplot(data=dataframe, x=var, y=y, ax=ax, scatter_kws={'alpha': 0.6})
        elif plot_type == 'box':
            sns.boxplot(data=dataframe, x=var, ax=ax)

    # Ensure the input data is sufficient
    if plot_type == 'reg' and y is None:
        raise ValueError("Please provide a dependent variable (y) for regplot.")

    # Set a clean theme
    sns.set_theme(style="whitegrid")

    variable_number = dataframe.shape[1]
    ncol = 5
    nrow = (variable_number - 1) // ncol + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 12))
    
    for var, ax in zip(dataframe.columns, axes.flatten()):
        _plot()
        if des_string:
            add_text(ax, stats_descriptive_string(dataframe[var]))

    # Remove any empty subplots
    for ax in axes.flatten()[variable_number:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
    

# feature processing
def _domain_professional_knowledge():
    print("""Before using any numerical methods to filter features,
          please use theories to explain what kind of variables should we 
          concern more and what should be less important
          """)

def filter_by_partial_linear(dataframe, target_variable="Desired_Savings",
                             lower_threshold=0.4, upper_threshold=0.9):
    """
    Filter columns based on their partial linear correlation coefficients
    with the target variable.

    Args:
        dataframe (pd.DataFrame): DataFrame with all numerical values.
        target_variable (str): The target variable to calculate correlations against.
        lower_threshold (float): Minimum value of the partial linear coefficient to keep.
        upper_threshold (float): Maximum value of the partial linear coefficient to keep.

    Returns:
        dataframe: the filtered dataframe
    """
    # Calculate correlation coefficients with the target variable
    partial_linear_coeff = dataframe.corr()[target_variable]

    # Filter columns meeting the threshold criteria
    filtered_columns = partial_linear_coeff[
        (partial_linear_coeff >= lower_threshold) & 
        (partial_linear_coeff <= upper_threshold)
    ].index.tolist()
    
    # include the three string variables
    filtered_columns.append("Occupation")
    filtered_columns.append("City_Tier")
    
    dataframe = dataframe[filtered_columns]

    return dataframe

# to the filterd dataframe, use Pipemethod to cleaning and transform the data
# 1. Impute the missing values
impute_missing = ColumnTransformer([
    ("num_", KNNImputer(n_neighbors = 5, weights = 'distance'), make_column_selector(dtype_include = np.number))
    ,("cat_", SimpleImputer(strategy = 'constant', fill_value = 'Unknown'), make_column_selector(dtype_include = object))
])

# 2. transform the features
feature_transformer = ColumnTransformer([
        ("PTS", PowerTransformer(method = 'yeo-johnson'), make_column_selector(dtype_include = np.number)),
        # ('scaler', RobustScaler(), make_column_selector(dtype_include = np.number)), # new change, scale the data
        ('OneHot', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), make_column_selector(dtype_include = 'string'))
    ]
    )

def _transform_data(data, transformer):
    """Apply a pipline on a dataframe and return the transformed array
    back to a dataframe with corresponding column names

    Args:
        data (_type_): _description_
        transformer (_type_): _description_

    Returns:`
        _type_: _description_
    """
    transformed_array = transformer.fit_transform(data)
    column_names = transformer.get_feature_names_out()
    # Convert to a pandas DataFrame
    transformed_df = pd.DataFrame(transformed_array, columns=column_names)
    # update the date types in the dataframe
    transformed_df = transformed_df.convert_dtypes()
    # Output the DataFrame
    return transformed_df

def pipe_line_transfer(dataframe, *transformers):
    """Apply the pipline of transfomration on the given dataframe
    and return the corresponding dataframe.

    Args:
        dataframe (_type_): the dataframe that have been suitably transformed
    """
    for transformer in transformers:
        dataframe = _transform_data(dataframe, transformer)
    
    return dataframe

# detect the outliers in a dataframe
def detect_observations_iqr(series, lower_quantil=0.25, upper_quantil=0.75, type='outlier', length_of_fence=0):
    """
    Detect observations in a pandas Series based on the Interquartile Range (IQR) method.
    
    Parameters:
        series (pd.Series): Input data series to analyze.
        lower_quantil (float): Lower quantile threshold (default is 0.25 for Q1).
        upper_quantil (float): Upper quantile threshold (default is 0.75 for Q3).
        type (str): Type of observations to detect. 
                    Use 'outlier' to find outliers or 'inlier' to find inliers. Default is 'outlier'.
        length_of_fence (float): Multiplier for the IQR to extend the fence for outlier/inlier detection. Default is 0.

    Returns:
        pd.Series: A pandas Series containing the detected outliers or inliers.
        tuple: A tuple containing the lower and upper bounds used for detection.
    """
    # Calculate Q1, Q3, and IQR
    Q1 = series.quantile(lower_quantil)
    Q3 = series.quantile(upper_quantil)
    IQR = Q3 - Q1

    # Define bounds for outliers/inliers
    lower_bound = Q1 - length_of_fence * IQR
    upper_bound = Q3 + length_of_fence * IQR

    # Detect outliers or inliers
    if type == 'outlier':
        observations = series[(series < lower_bound) | (series > upper_bound)]
    elif type == 'inlier':
        observations = series[(series >= lower_bound) & (series <= upper_bound)]
    else:
        raise ValueError("Parameter 'type' must be either 'outlier' or 'inlier'.")

    return observations


def filter_df_by_quantile(dataframe, column, lower_quantile=0.25, upper_quantile=0.75, type='outlier', length_of_fence= 0):
    """
    Filter rows in a DataFrame based on the quantiles of a specified column.
    
    Parameters:
        dataframe (pd.DataFrame): The input DataFrame to filter.
        column (str): The name of the column to analyze for quantiles.
        lower_quantile (float): Lower quantile threshold (default is 0.25 for Q1).
        upper_quantile (float): Upper quantile threshold (default is 0.75 for Q3).
        type (str): Type of rows to retain based on the quantiles.
                    Use 'outlier' to keep rows with outliers or 'inlier' to keep rows with inliers. Default is 'outlier'.

    Returns:
        pd.DataFrame: A filtered DataFrame containing rows matching the criteria.
    """
    # Detect relevant observations based on quantiles
    filtered_series = detect_observations_iqr(dataframe[column], lower_quantile, upper_quantile, type, length_of_fence)

    # Filter the DataFrame rows based on the detected observations
    filtered_df = dataframe.loc[filtered_series.index, :]

    return filtered_df

def introduce_missing_values(dataframe, missing_fraction=0.1, random_seed=None):
    """
    Randomly remove values from a DataFrame to introduce missing values.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        missing_fraction (float): Fraction of values to be removed (default is 0.1).
                                  Must be between 0 and 1.
        random_seed (int): Random seed for reproducibility (default is None).

    Returns:
        pd.DataFrame: A new DataFrame with missing values introduced.
    """
    if not 0 <= missing_fraction <= 1:
        raise ValueError("missing_fraction must be between 0 and 1.")
    
    # Convert DataFrame to float to handle NaN
    df_with_missing = dataframe.copy()
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate the total number of values and the number of missing values
    total_values = df_with_missing.size
    num_missing = int(total_values * missing_fraction)
    
    # Randomly select indices for missing values
    missing_indices = np.random.choice(total_values, num_missing, replace=False)
    
    # Flatten the DataFrame and set selected indices to NaN
    flat_values = df_with_missing.values.flatten()
    flat_values[missing_indices] = np.nan
    
    # Reshape the flattened array back to the original DataFrame shape
    df_with_missing = pd.DataFrame(flat_values.reshape(dataframe.shape), 
                                   columns=dataframe.columns, 
                                   index=dataframe.index)
    
    return df_with_missing
    
# fit some base models with default parameters to find a relatively good one
# Create base models
def base_model_list(basemodels = None):
    """Create a list in which the elements are tuples in ('name', <estimator>)
    form. There are six benchmark models to compare.
    

    Returns:
        basemodel(list): list of tuple of '('name', <estimator>)' form, added as an estimator
        to be compared with the benchmarks
    """
    base_models = [
        
                    ("LinearRegression", LinearRegression()),
                    ("Ridge", Ridge()),
                    ("ElasticNet", ElasticNet()),
                    ("DecisionTreeRegressor", DecisionTreeRegressor()),
                    ("SVR", SVR()),
                    ("RandomForestRegressor", RandomForestRegressor()),
                ]
    if basemodels == None:
        pass
    else:
        base_models.extend(basemodels)
    return base_models

# Define custom RMSE scorer
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Define custom log-likelihood scorer
def log_likelihood(y_true, y_pred):
    residual = y_true - y_pred
    sigma2 = np.var(residual)
    n = len(y_true)
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return ll

loglik_scorer = make_scorer(log_likelihood, greater_is_better=True)



def cross_validate_bench(X, y, base_models = None, fold_times = 5, add_basemodels = None, scorer = None):
    """
    Assess the performance of given models with default parameter by cross-validate method.
    Used to check the simple performance on different models.

    Args:
        X (_type_): _description_
        y (_type_): _description_
        base_models (_type_): _description_
        fold_times (int, optional): _description_. Defaults to 5.
        add_basemodels (_type_, optional): _description_. Defaults to None.
        scorer (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Scoring dictionary
    if scorer == None:
        scoring = {
            'MSE': 'neg_mean_squared_error',  # Built-in scorer
            'R2': 'r2',                       # Built-in scorer
            'RMSE': rmse_scorer,              # Custom scorer
            'Log-Lik': loglik_scorer,          # Custom scorer
        }
    else:
        scorer = scorer

    base_models = base_model_list(add_basemodels)
    # Model
    fold_times = fold_times

    model_name = []
    model_mean_test_MSE = []
    model_mean_test_RMSE = []
    model_mean_test_R2 = []
    model_mean_test_loglik = []

    # Cross-validate
    for name, model in base_models:
        scores = cross_validate(model,
                            X, y, 
                            cv = fold_times,
                            scoring=scoring)
        model_name.append(name)
        
        model_mean_test_MSE.append(-np.mean(scores["test_MSE"]))
        model_mean_test_RMSE.append(-np.mean(scores["test_RMSE"]))
        model_mean_test_R2.append(np.mean(scores["test_R2"]))
        model_mean_test_loglik.append(np.mean(scores["test_Log-Lik"]))

    models_cross_val_results_df = pd.DataFrame(
    {"Model": model_name,
     "Mean_Test_MSE": model_mean_test_MSE,
     "Mean_Test_RMSE": model_mean_test_RMSE,
     "Mean_Test_R2": model_mean_test_R2,
     "Mean_Test_Log-Lik": model_mean_test_loglik}).round(4)
    
    return models_cross_val_results_df

def _new_col_name(old_col_name):
    """Transform the old column name to a new one, due to the Pipeline transformation
    on column values and names.

    Args:
        old_col_name (_type_): string, the old column names.
    """
    
    return "PTS__num___" + old_col_name

# Define models and grid searching settings for finding the best model
def grid_search_settings(models = None, param_grid = None, scorer = None):
    """Generate the parameters for GridSearchCV function to find the best model
     and its parameters under the given grid
     
     models(dict): dictionary that stores the name of model and the instance as the 
     value
     param_grids(dict): dictionary with fied of the model name and spcific parameter values
     used to span the grid space
     scorer(sklearn score instance): the criterion used to reflect the performance of models
     """
    if models == None:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
        }

    # Define parameter grids
    if param_grid == None:
        param_grids = {
            "LinearRegression": {
                "fit_intercept": [True, False],
                "positive": [True, False],  # If normalize is not deprecated in your Scikit-learn version
            },
            "RandomForestRegressor": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
        }

    if scorer == None:
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    grid_search_dict = {"models": models,
                        "param_grids": param_grids,
                        "scorer": scorer}
    
    return grid_search_dict

def Grid_Search_CV(X, y, _grid_search_settings):
    """Instance a GridSearchCV instance, fit and tranform it on the given dataframe
    
    """
    models = _grid_search_settings["models"]
    param_grids = _grid_search_settings["param_grids"]
    scorer = _grid_search_settings["scorer"]
    best_models = {}
    for model_name, model in models.items():
        print(f"Running GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring=rmse_scorer,
        cv=5,  # Number of folds in cross-validation
        verbose=2,
        n_jobs=-1,  # Use all available cores
    )
        grid_search.fit(X,y)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score for {model_name}: {grid_search.best_score_}")
    
    return best_models
        
        
    
    

if __name__ == "__main__":
    # read data
    data_store_path = '/Users/gufeng/2024_Fall/dasc6810/6810 Final project/data'
    sample_data = read_data()

    # split the data set into edging 20% and central 80%
    lower_quantile = 0.1
    upper_quantile = 0.9
    central_80_data = filter_df_by_quantile(sample_data, "Desired_Savings", lower_quantile, upper_quantile, 'inlier')
    central_80_data_transformed = central_80_data.copy()
    central_20_data = filter_df_by_quantile(sample_data, "Desired_Savings", lower_quantile, upper_quantile, 'outlier')
    central_20_data_transformed = central_20_data.copy()
    
    print(central_80_data_transformed.columns)

    # make feature transformation on the two data sets
    for transformer in [impute_missing, feature_transformer]:
        central_80_data_transformed = _transform_data(central_80_data_transformed, transformer)
        central_20_data_transformed = _transform_data(central_20_data_transformed, transformer)
    
    # store the transformed data sets to local
    central_80_data_transformed.to_csv(data_store_path + '/central_80_data_transformed.csv')
    central_20_data_transformed.to_csv(data_store_path + '/central_20_data_transformed.csv')

    # build base models on one set to check the performance of models 
    # the column names have changeed due to the feature transformation
    X = central_80_data_transformed.drop(columns = _new_col_name("Desired_Savings"))
    y = central_80_data_transformed[_new_col_name('Desired_Savings')]
    cross_validate_results_df = cross_validate_bench(X, y)
    print(cross_validate_results_df.head())
    
    # after check the result from cross_vakudate_results, we choose LinearRegression
    # and RandomForestRegressor as the two best models and apply GridSearchCV method on
    # them to find the two with best parameters
    # * running time ~ 1.5mintue *
    best_models = Grid_Search_CV()
    
    
    
    
    # print(sample_data.head())
    # print(check_dataframe_info(sample_data))
    # _plot_numerical_distribution(sample_data, plot_type='reg', y = "Desired_Savings")
    # print(filter_by_partial_linear(sample_data).head())
    
    # transform the original data in a pipeline
    # transformed_data = sample_data.copy()
    # for transformer in [impute_missing, feature_transformer]:
    #     transformed_data = _transform_data(transformed_data, transformer)
    # print(transformed_data.head())
    # print(filter_df_by_quantile(sample_data, "Income", 0.1, 0.9, 'inlier').head())
    
    