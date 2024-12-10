import time 
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
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.stats as stats



# read data from local
def read_data(type = 'sample'):
    """Read data set into a dataframe

    Args:
        type (str, optional): 
        Defaults to 'sample', reading the sample data(2000,27),
        if 'complete_train', then read the complete train dataset of (18000, 27)
        if 'complete_test', then read the complete test dataset of (2000,27)
    """
    data_path = "/Users/gufeng/2024_Fall/dasc6810/6810 Final project/data"
    if type == 'sample':
        data = pd.read_csv(data_path + "/sampled_df_trian.csv", index_col=0)
    elif type == 'complete_train':
        data = pd.read_csv(data_path + "/complete_train_data.csv").drop("Unnamed: 0", axis = 1)
    elif type == 'complete_test':
        data = pd.read_csv(data_path + "/complete_test_data.csv").drop("Unnamed: 0", axis = 1)

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

document_image_path = '/Users/gufeng/2024_Fall/dasc6810/6810 Final project/documents/images'

def _save_plot(fig, pre_fix, director_path = document_image_path, dpi = 100):
    """Save the current fig object from matplotlib

    Args:
        director_path (string): _description_
        pre_fix (string): _description_
    """
    if pre_fix != None:
        fig.savefig(f"{director_path}/{pre_fix}.png", dpi = dpi)
    else:
        fig.savefig(f"{director_path}/_nan_.png", dpi = dpi)
        

# plots function
def _plot_numerical_distribution(dataframe, plot_type='hist', pre_fix = "None", dpi = 300, figsize = (15, 12), des_string=False, y=None):
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
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    
    for var, ax in zip(dataframe.columns, axes.flatten()):
        _plot()
        if des_string:
            add_text(ax, stats_descriptive_string(dataframe[var]))

    # Remove any empty subplots
    for ax in axes.flatten()[variable_number:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    
    _save_plot(fig, pre_fix, document_image_path, dpi = dpi)
    

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
    # include the transormed one-hot variables
    old_columns = dataframe.columns.values
    filtered_columns.extend(old_columns[-5:-1])
    filtered_columns.append(target_variable)


    return dataframe[filtered_columns] # only use this function after transformed the columns

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
    # Base model list
    base_models = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge()),
        ("ElasticNet", ElasticNet()),
        ("Lasso", Lasso()),
        ("DecisionTreeRegressor", DecisionTreeRegressor()),
        ("SVR", SVR()),
        ("RandomForestRegressor", RandomForestRegressor()),
        ("GradientBoostingRegressor", GradientBoostingRegressor()),
        ("AdaBoostRegressor", AdaBoostRegressor()),
        ("KNeighborsRegressor", KNeighborsRegressor()),
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
            "positive": [True, False],  # Enforce non-negative coefficients
            "n_jobs": [-1, None],  # Parallel computation
        },
        "RandomForestRegressor": {
            "n_estimators": [50, 100, 200],  # Number of trees
            "max_depth": [20, 30],  # Tree depth (None for unlimited depth)
            "min_samples_split": [5, 10,],  # Minimum samples required to split a node
            "min_samples_leaf": [1, 4, 6],  # Minimum samples required at each leaf node
            "criterion": ["squared_error"],  # Criterion for measuring quality of a split
            "n_jobs": [-1]
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
        n_jobs= -1,  # Use all available cores
    )
        grid_search.fit(X,y)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score for {model_name}: {grid_search.best_score_}")
    
    return best_models

# draw the predictions from the best models on both train and test datasets

def visualize_train_test_predictions(models, X_train, y_train, X_test, y_test, save_path=None):
    """
    Visualize fitted values for training data and predictions for test data, with an option to save the plot.
    
    Parameters:
    - models: dict, containing fitted models with their names as keys
    - X_train: pd.DataFrame or np.array, training predictors
    - y_train: pd.Series or np.array, training target values
    - X_test: pd.DataFrame or np.array, test predictors
    - y_test: pd.Series or np.array, test target values
    - save_path: str, optional, file path to save the plot (e.g., "output/plot.png")
    
    Returns:
    - predictions: dict, containing test predictions for each model
    """
    predictions = {}
    fig, axes = plt.subplots(2, len(models), figsize=(14, 10), constrained_layout=True)
    
    # Iterate through models
    for idx, (model_name, model) in enumerate(models.items()):
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict on training and test data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        predictions[model_name] = y_pred_test  # Store test predictions
        
        # Plot training fitted values
        axes[0, idx].scatter(y_train, y_pred_train, alpha=0.6, label='Fitted Values')
        axes[0, idx].plot(
            [min(y_train), max(y_train)], 
            [min(y_train), max(y_train)], 
            color='red', linestyle='--', label='Ideal Fit'
        )
        axes[0, idx].set_title(f"{model_name} - Train")
        axes[0, idx].set_xlabel("Actual Values")
        axes[0, idx].set_ylabel("Predicted Values")
        axes[0, idx].legend()
        axes[0, idx].grid()
        
        # Plot test predictions
        axes[1, idx].scatter(y_test, y_pred_test, alpha=0.6, label='Predicted Values')
        axes[1, idx].plot(
            [min(y_test), max(y_test)], 
            [min(y_test), max(y_test)], 
            color='red', linestyle='--', label='Ideal Fit'
        )
        axes[1, idx].set_title(f"{model_name} - Test")
        axes[1, idx].set_xlabel("Actual Values")
        axes[1, idx].set_ylabel("Predicted Values")
        axes[1, idx].legend()
        axes[1, idx].grid()
    
    # Save the plot if save_path is provided
    save_path = document_image_path + "/predictions_on_train_test"
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
# draw the residual diagonistic plots for the two best models
def plot_residual_diagnostics_combined(models, X, y, save_path=None):
    """
    Plots residual diagnostics for all models in one figure for comparison and optionally saves the plot.

    Parameters:
        models (dict): Dictionary of fitted models (name -> model).
        X (DataFrame): Training predictor variables.
        y (Series or array-like): Training target variable.
        save_path (str): Path to save the plot (optional). If None, plot is not saved.
    """
    n_models = len(models)
    fig, axs = plt.subplots(n_models, 2, figsize=(12, 5 * n_models))  # Create subplots: n_models x 2
    fig.suptitle("Residual Diagnostics for All Models", fontsize=18)

    for i, (model_name, model) in enumerate(models.items()):
        # Get predicted and residual values
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Ensure residuals are a NumPy array (prevent single-value issues)
        residuals = np.array(residuals)

        # Histogram of residuals
        sns.histplot(residuals, kde=True, ax=axs[i, 0], color="skyblue")
        axs[i, 0].set_title(f"{model_name}: Histogram of Residuals")
        axs[i, 0].set_xlabel("Residuals")
        axs[i, 0].set_ylabel("Frequency")

        # Residuals vs fitted values
        axs[i, 1].scatter(y_pred, residuals, alpha=0.6, color="orange")
        axs[i, 1].axhline(0, color="red", linestyle="--")
        axs[i, 1].set_title(f"{model_name}: Residuals vs Fitted")
        axs[i, 1].set_xlabel("Fitted Values")
        axs[i, 1].set_ylabel("Residuals")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
        file_name = "combined_residual_diagnostics.png"
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {full_path}")        
    
    

if __name__ == "__main__":
    # read data
    data_store_path = '/Users/gufeng/2024_Fall/dasc6810/6810 Final project/data'
    train_data = read_data(type="complete_train")
    test_data = read_data(type = 'complete_test')

    # save the box plot for taw data
    # _plot_numerical_distribution(sample_data, 'box', "box-plot_raw_data")
    
    # split the data set into edging 20% and central 80%
    lower_quantile = 0.1
    upper_quantile = 0.9
    central_80_data_train = filter_df_by_quantile(train_data, "Desired_Savings", lower_quantile, upper_quantile, 'inlier')
    central_80_data_train_transformed = central_80_data_train.copy()
    central_80_data_test = filter_df_by_quantile(test_data, "Desired_Savings", lower_quantile, upper_quantile, 'inlier')
    central_80_data_test_transformed = central_80_data_test.copy()

    
     # make feature transformation on the two data sets
    for transformer in [impute_missing, feature_transformer]:
        central_80_data_train_transformed = _transform_data(central_80_data_train_transformed, transformer)
        central_80_data_test_transformed = _transform_data(central_80_data_test_transformed, transformer)
        
    
    # save the box plot for transformed 80% data
    # _plot_numerical_distribution(central_80_data_train_transformed, 'box', "box-plot_tansformed_data")   
    
    # # store the transformed data sets to local
    central_80_data_train_transformed.to_csv(data_store_path + '/central_80_data_train_transformed.csv')
    central_80_data_test_transformed.to_csv(data_store_path + '/central_80_data__test_transformed.csv')
    
    filtered_central_80_data_train_transformed = \
    filter_by_partial_linear(central_80_data_train_transformed, 
                             _new_col_name("Desired_Savings"), 
                             lower_threshold = 0.4, 
                             upper_threshold = 0.8)
    
    filtered_central_80_data_test_transformed = central_80_data_test_transformed[filtered_central_80_data_train_transformed.columns.values]
    
    # filtered_central_80_data_transformed.to_csv(data_store_path + "/column_filtered_80_data_transformed.csv")

    # Initial check on the 10 models
    # build base models on one set to check the performance of models 
    # the column names have changeed due to the feature transformation
    X = filtered_central_80_data_train_transformed.drop(columns = _new_col_name("Desired_Savings"))
    y = filtered_central_80_data_train_transformed[_new_col_name('Desired_Savings')]
    
    X_test = filtered_central_80_data_test_transformed.drop(columns = _new_col_name("Desired_Savings"))
    y_test = filtered_central_80_data_test_transformed[_new_col_name('Desired_Savings')]
    
    # # time this searching work
    timing_record = {}
    start_time = time.time()
    cross_validate_results_df = cross_validate_bench(X, y)
    cross_validate_results_df.to_csv(data_store_path + '/cross_validate_model_results.csv')
    end_time = time.time()
    elapsed_time = end_time - start_time
    timing_record["work"] = ["cross_vlidate_10_models"]
    timing_record["time"] = [elapsed_time]
    
    print(cross_validate_results_df.head())
    print(cross_validate_results_df.shape)
    print("###" * 10)
    print("The initial check on 10 models is finished.")
    
    # # after check the result from cross_vakudate_results, we choose LinearRegression
    # # and RandomForestRegressor as the two best models and apply GridSearchCV method on
    # # them to find the two with best parameters
    
    # # time the Grid Search work
    start_time = time.time()
    best_models = Grid_Search_CV(X, y, grid_search_settings())
    print("###" * 10)
    print("The Grid search work is finished.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    timing_record["work"].append("grid_search_parameters")
    timing_record["time"].append(elapsed_time)
    
    # store the timing dictionary
    timing_record = pd.DataFrame(timing_record)
    timing_record.to_csv(data_store_path + "/work_timing.csv")
    
    # store the best models information to a csv file
    best_models_df = pd.DataFrame.from_dict(best_models, orient="index", columns=["Details"]).reset_index()
    # Rename columns
    best_models_df.columns = ["Model", "Details"]
    # store it
    best_models_df.to_csv(data_store_path + "/best_models_info.csv")
    
    # visualize the predictions on train and test datasets
    visualize_train_test_predictions(best_models, X, y, X_test, y_test)
    # visualize the residuals for predictions on train and test dataset
    plot_residual_diagnostics_combined(best_models, X, y, 
                              save_path = document_image_path
                              )
    plot_residual_diagnostics_combined(best_models, X_test, y_test, 
                              save_path = document_image_path
                              )
    
    