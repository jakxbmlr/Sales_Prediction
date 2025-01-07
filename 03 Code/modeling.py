import pandas as pd # type: ignore
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model # type: ignore (unnecessary import resolve warning)
from tensorflow.keras.models import load_model as keras_load_model # type: ignore
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization, Activation, Input, MultiHeadAttention, LayerNormalization, Dropout, Reshape, Concatenate, Lambda # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, ReduceLROnPlateau # type: ignore
from tensorflow.keras.saving import register_keras_serializable # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.losses import MeanSquaredLogarithmicError, Huber  # type: ignore
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
import plotly.graph_objects as go
import plotly.io as pio
import chart_studio.plotly as py
from dateutil.relativedelta import relativedelta
import shutil
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import shap
from data_analysis import create_plot, color_palette
from data_integration import find_largest_subseries

#pio.renderers.default = "vscode" # pio.renderers.default = "notebook_connected"

# PREPROCESSING
class Dataset:
    def __init__(self, data):
        # Imported raw data
        self.target_df = data[0]
        self.feature_df = data[1]
        
        # Names of target and feature attributes/columns
        self.time_series = data[2]
        self.target_single = data[3]
        self.features = data[4]
        self.targets_multi = data[5]
        
        # Scaler
        self.feature_scaler = data[6]
        self.target_scaler = data[7]

        # Features (X), Targets (y) and indices (dates) over the whole time span for multi and single output respectively
        self.X_multi_all = data[8]
        self.y_multi_all = data[9]
        self.dates_multi_all = data[10]
        self.X_single_all = data[11]
        self.y_single_all = data[12]
        self.dates_single_all = data[13]

        # Features (X), Targets (y) and indices (dates) (inner join on X and y) for multi and single output respectively
        self.X_single = data[14]
        self.y_single = data[15]
        self.dates_single = data[16]
        self.X_multi = data[17]
        self.y_multi = data[18]
        self.dates_multi = data[19]

        # Features (X), Targets (y) and indices (dates) seperated into training, validation and test sets for multi and self.single output respectively + reshaped Inputs for models that only support 1-dimensional input vectors
        self.X_train_single = data[20]
        self.X_val_single = data[21]
        self.X_test_single = data[22]
        self.y_train_single = data[23]
        self.y_val_single = data[24]
        self.y_test_single = data[25]

        self.dates_train_single = data[26]
        self.dates_val_single = data[27]
        self.dates_test_single = data[28]

        self.X_train_single_reshaped = data[29]
        self.X_val_single_reshaped = data[30]
        self.X_test_single_reshaped = data[31]

        self.X_train_multi = data[32]
        self.X_val_multi = data[33]
        self.X_test_multi = data[34]
        self.y_train_multi = data[35]
        self.y_val_multi = data[36]
        self.y_test_multi = data[37]

        self.dates_train_multi = data[38]
        self.dates_val_multi = data[39]
        self.dates_test_multi = data[40]

        self.X_train_multi_reshaped = data[41]
        self.X_val_multi_reshaped = data[42]
        self.X_test_multi_reshaped = data[43]
def load_preprocess_form_dataset(feature_files: list[str], target_file: str) -> tuple:
    # Load Data
    hg_oi_df = pd.read_csv(target_file, parse_dates=['TIME_PERIOD'], index_col='TIME_PERIOD')
    
    # Assuming feature_files contains paths to the feature data CSVs
    feature_dfs = [pd.read_csv(file, parse_dates=['TIME_PERIOD'], index_col='TIME_PERIOD') for file in feature_files]
    feature_dfs = [df.drop('index', axis=1) if "index" in df.columns else df for df in feature_dfs ]
    
    # Concatenate all feature dataframes along the columns
    feature_df = pd.concat(feature_dfs, axis=1)

    # Include month feature (delete to work with V2 Models)
    feature_df.index = pd.to_datetime(feature_df.index)
    feature_df["Month"] = feature_df.index.month

    month_dummies = pd.get_dummies(feature_df["Month"], prefix="Month").astype(int)
    feature_df = feature_df.drop("Month", axis=1)
    feature_df = pd.concat([month_dummies, feature_df], axis=1)

    feature_df = feature_df[month_dummies.columns.tolist() + feature_df.columns.drop(month_dummies.columns).tolist()]

    # Preprocessing
    check_data(feature_df)
    preprocessed_feature_df = feature_df.apply(preprocess_series)
    # preprocessed_oecd_df.reset_index(inplace=True, drop=False)
    # preprocessed_oecd_df, properties = find_largest_subseries(preprocessed_oecd_df, max_consecutive_invalids=20)
    # preprocessed_oecd_df.set_index("TIME_PERIOD")
    feature_df_cleaned = drop_columns_with_missing_data(preprocessed_feature_df)

    # Align and clean data
    feature_df_cleaned, hg_oi_df_aligned = align_and_clean_data(hg_oi_df, feature_df_cleaned)

    # Feature and target names
    target = "T"
    time_series = [f"OI_{target}"] + [f"{col}" for col in feature_df_cleaned.columns]
    target_single = [f"OI_1"]
    features = [[f"{col}_{m}" for col in time_series] for m in range(12)]
    targets_multi = [f"OI_{m}" for m in range(12)]

    # Scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    features_scaled = feature_scaler.fit_transform(feature_df_cleaned.values)
    oi_scaled = target_scaler.fit_transform(hg_oi_df_aligned[target].values.reshape(-1, 1))

    # Generate training data
    X_single, y_single, dates_single = create_training_data_single_target(oi_scaled, features_scaled, hg_oi_df_aligned.index)
    X_multi, y_multi, dates_multi = create_training_data_multi_target(oi_scaled, features_scaled, hg_oi_df_aligned.index)

    # Generate dataset with all data for predictions
    X_multi_all, y_multi_all, dates_multi_all = create_all_data_multi_target(oi_scaled, features_scaled, hg_oi_df_aligned.index)
    X_single_all, y_single_all, dates_single_all = create_all_data_multi_target(oi_scaled, features_scaled, hg_oi_df_aligned.index)

    # Split and reshape data for single target
    X_train_single, X_val_single, X_test_single, y_train_single, y_val_single, y_test_single, dates_train_single, dates_val_single, dates_test_single, X_train_single_reshaped, X_val_single_reshaped, X_test_single_reshaped = split_and_reshape_data_chronologically(X_single, y_single, dates_single, test_size=0.15, val_size=0.15)
    
    # For Linear Regression and XGBoost, no validation data needed
    X_train_single_reshaped = np.concatenate((X_train_single_reshaped, X_val_single_reshaped), axis=0)
    y_train_single = np.concatenate((y_train_single, y_val_single), axis=0)

    # Split and reshape data for multiple targets
    X_train_multi, X_val_multi, X_test_multi, y_train_multi, y_val_multi, y_test_multi, dates_train_multi, dates_val_multi, dates_test_multi, X_train_multi_reshaped, X_val_multi_reshaped, X_test_multi_reshaped = split_and_reshape_data_chronologically(X_multi, y_multi, dates_multi, test_size=0.15, val_size=0.15)

    return (hg_oi_df, feature_df, # Imported raw data
            time_series, target_single, features, targets_multi, # name of target and feature attributes/columns

            # Scaler
            feature_scaler,
            target_scaler,
            
            # Features (X), Targets (y) and indices (dates) over the whole time span (outter join on X) for multi and single output respectively
            X_multi_all, y_multi_all, dates_multi_all,
            X_single_all, y_single_all, dates_single_all,

            # Features (X), Targets (y) and indices (dates) (inner join on X and y) for multi and single output respectively
            X_single, y_single, dates_single,
            X_multi, y_multi, dates_multi,

            # Features (X), Targets (y) and indices (dates) seperated into training, validation and test sets for multi and single output respectively + reshaped Inputs for models that only support 1-dimensional input vectors
            X_train_single, X_val_single, X_test_single, y_train_single, y_val_single, y_test_single, 
            dates_train_single, dates_val_single, dates_test_single, 
            X_train_single_reshaped, X_val_single_reshaped, X_test_single_reshaped,
            X_train_multi, X_val_multi, X_test_multi, y_train_multi, y_val_multi, y_test_multi, 
            dates_train_multi, dates_val_multi, dates_test_multi, 
            X_train_multi_reshaped, X_val_multi_reshaped, X_test_multi_reshaped)

def normalize_series(series, method='standard'):
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(series_scaled, index=series.index)
def preprocess_series(series, max_consecutive_nans=3):
    series = series.replace('nan', np.nan).astype(float)
    series_interpolated = series.interpolate(method='linear', limit=max_consecutive_nans, limit_direction='both')
    is_not_nan = ~series_interpolated.isna()
    seg_change = (is_not_nan != is_not_nan.shift()).cumsum()
    valid_segments = [segment for _, segment in series_interpolated.groupby(seg_change) if segment.isna().sum() == 0]
    if not valid_segments:
        return pd.Series(dtype=series.dtype)
    preprocessed = valid_segments[-1]
    return preprocessed
def check_data(df):
    df.replace('nan', np.nan, inplace=True)
    null_values_summary = df.isnull().sum()
    print("Summary of null values in each column:")
    print(null_values_summary)
def drop_columns_with_missing_data(df, threshold=5):
    columns_to_drop = df.columns[df.isnull().sum() >= threshold]
    df_cleaned = df.drop(columns=columns_to_drop)
    print("Dropped columns with 5 or more missing months:")
    print(columns_to_drop)
    return df_cleaned
def align_and_clean_data(target_df, feature_df):
    aligned_index = target_df.index.intersection(feature_df.index)
    feature_df_cleaned = feature_df.loc[aligned_index]
    target_df_aligned = target_df.loc[aligned_index]
    feature_df_cleaned.dropna(inplace=True)
    target_df_aligned = target_df_aligned.loc[feature_df_cleaned.index]
    print("Shape of cleaned feature DataFrame:", feature_df_cleaned.shape)
    print("Shape of aligned target DataFrame:", target_df_aligned.shape)
    return feature_df_cleaned, target_df_aligned

def create_training_data_single_target(oi, features, dates, n_months=12):
    X, y, date_list = [], [], []
    for i in range(len(oi) - n_months):
        X.append(np.concatenate([oi[i:i+n_months], features[i:i+n_months]], axis=1))
        y.append(oi[i+n_months])
        date_list.append(dates[i+n_months])
    return np.array(X), np.array(y), np.array(date_list)
def create_training_data_multi_target(oi, features, dates, n_months=12):
    X, y, date_list = [], [], []
    for i in range(len(oi) - n_months - 11):
        X.append(np.concatenate([oi[i:i+n_months], features[i:i+n_months]], axis=1))
        y.append(oi[i+n_months:i+n_months+12].flatten())
        date_list.append(dates[i+n_months])
    return np.array(X), np.array(y), np.array(date_list)
def create_all_data_multi_target(oi, features, dates, n_months=12):
    X = [np.concatenate([oi[i:i+n_months], features[i:i+n_months]], axis=1) for i in range(len(oi) - n_months)]
    dates = [dates[i+n_months] for i in range(len(oi) - n_months)]
    y = [oi[i+n_months:i+n_months+12].flatten() if i+n_months+12 < len(oi) else np.pad(oi[i+n_months:len(oi)-1].flatten(), (0, 12 - (len(oi) - 1 - i - n_months)), 'constant') for i in range(len(oi) - n_months)]
    return np.array(X), np.array(y), np.array(dates)
def create_all_data_single_target(oi, features, dates, n_months=12):
    X, y, date_list = [], [], []
    for i in range(len(oi) - n_months + 1):
        X.append(np.concatenate([oi[i:i+n_months], features[i:i+n_months]], axis=1))
        y.append(oi[i+n_months] if i+n_months < len(oi) else None)
        date_list.append(dates[i+n_months])
    return np.array(X), np.array(y), np.array(date_list)
def split_and_reshape_data_chronologically(X, y, dates, test_size=0.2, val_size=0.2):
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    dates_train, dates_val, dates_test = dates[:n_train], dates[n_train:n_train+n_val], dates[n_train+n_val:]

    # Combine training and validation sets
    # X_train = np.concatenate((X_train, X_val), axis=0)
    # y_train = np.concatenate((y_train, y_val), axis=0)

    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test, X_train_reshaped, X_val_reshaped, X_test_reshaped
def extended_features(features):
    return [f"{name}_{i}" for name in features for i in range(12)]


# TRAINING

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
def history_to_json(history, filepath):
    history_dict = {key: [convert_to_json_serializable(val) for val in values] for key, values in history.items()}
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f)
def lr_scheduler_function(epoch, lr, decay_rate=0.6, decay_step=10):
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
class DynamicBatchSizeCallback(Callback):
    def __init__(self, initial_batch_size, decay_factor, decay_epoch):
        self.batch_size = initial_batch_size
        self.decay_factor = decay_factor
        self.decay_epoch = decay_epoch
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and epoch % self.decay_epoch == 0:
            self.batch_size = max(1, int(round(self.batch_size / self.decay_factor)))
            print(f"Reducing batch size to: {self.batch_size}")

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['batch_size'] = self.batch_size
class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Depending on TensorFlow version, use one of these:
        lr = self.model.optimizer.learning_rate.numpy()
        #lr = tf.keras.backend.get_value(self.model.optimizer._decayed_lr(tf.float32))
        logs['lr'] = lr
# Linear Regression (Deprecated, use MultivariateEnsemble instead)
def build_and_train_lr_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def train_and_evaluate_lr_model(X_train, y_train, X_test, y_test, dates_test, scaler):
    linear_model, _ = build_and_train_lr_model(X_train, y_train)
    results_df = evaluate_model_single_target(linear_model, X_test, y_test, dates_test, scaler, model_name="Linear Regression")
    #results = {"MSE": mse_direct, "MAE": mae_rescaled}
    return linear_model, results_df
# XGBoost (Deprecated, use MultivariateEnsemble instead)
def build_and_train_xgb_model(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model, None
def train_and_evaluate_xgb_model(X_train, y_train, X_test, y_test, dates_test, scaler):
    xgb_model, _ = build_and_train_xgb_model(X_train, y_train)
    results_df = evaluate_model_single_target(xgb_model, X_test, y_test, dates_test, scaler, model_name="XGBoost")
    #results = {"MSE": mse_direct, "MAE": mae_rescaled}
    return xgb_model, results_df
# MLP
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Saved best model at: {model_path}")
def load_model(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model
class MLP_hyperparameters:
    def __init__(self, 
                hidden_layer_sizes=(100, 100, 100),
                max_iter=10000,
                learning_rate_init=0.00001,
                alpha=0.0001,
                solver='adam',
                validation_fraction=0.2,
                n_iter_no_change=20000):
        self.hidden_layer_sizes=hidden_layer_sizes
        self.max_iter=max_iter
        self.learning_rate_init=learning_rate_init
        self.alpha=alpha
        self.solver=solver
        self.validation_fraction=validation_fraction
        self.n_iter_no_change=n_iter_no_change
    def get_dict(self):
        return {key: value for key, value in self.__dict__.items()}
def train_and_evaluate_mlp_model(X_train, y_train, X_val, y_val, X_test, y_test, dates_test, scaler, model_name, dataset_name, hyperparameters=MLP_hyperparameters()):
    mlp = MLPRegressor(
        hidden_layer_sizes=hyperparameters.hidden_layer_sizes,
        max_iter=hyperparameters.max_iter,
        learning_rate_init=hyperparameters.learning_rate_init,
        alpha=hyperparameters.alpha,
        solver=hyperparameters.solver,
        validation_fraction=hyperparameters.validation_fraction,
        n_iter_no_change=hyperparameters.n_iter_no_change
    ).fit(
        np.concatenate((X_train, X_val), axis=0), 
        np.concatenate((y_train, y_val), axis=0)
    )
    
    model_save_path = os.path.join("Models", f"{model_name}.pkl")
    archive_path = os.path.join("Model Archive", f"{model_name}.pkl")
    model_paths = {
        f"{model_name} Archived": archive_path,
        f"{model_name}": model_save_path
    }
    save_model(mlp, model_save_path)

    evaluation_df = evaluate_and_save_best_model(model_paths, X_test, y_test, dates_test, scaler, archive_path, sklearn=True)
    # metrics = {
    #     "MSE": evaluation_df[evaluation_df["Model"]==model_filename]["MSE (scaled)"],
    #     "MAE": evaluation_df[evaluation_df["Model"]==model_filename]["MAE (rescaled)"]
    # }
    update_and_save_metrics(evaluation_df[evaluation_df["Model"] == model_name].copy(), dataset_name, hyperparameters=hyperparameters.get_dict(), excel_dir=".", file_name='training_results.xlsx')
    return mlp, evaluation_df
def load_and_evaluate_mlp_model_multi(model_path, X, y, dates, target_scaler):
    mlp = load_model(model_path)
    mse_direct, mae_rescaled, results_df = evaluate_model_multi_target(mlp, X, y, dates, target_scaler, model_name="MLP (multi output)")
    return mlp, results_df
# MLP (multi-output) HP-tuning with GridSearch
def train_and_evaluate_mlp_with_grid_search(X_train, y_train, X_val, y_val, X_test, y_test, dates_test, scaler, model_name, dataset_name):
    parameters = {
        'hidden_layer_sizes': [
            (100, 100), (100, 100, 100), (150, 100, 50)
        ],
        'max_iter': [10000],
        'learning_rate_init': [1e-5, 5e-6, 5e-5],
        'batch_size': [16, 32, "auto"],
        'learning_rate': ["adaptive"],
        'alpha': [0.01, 0],
        'solver': ['adam'],
        'validation_fraction': [0.2],
        'n_iter_no_change': [100],
        'random_state': [72],
        'tol': [1e-4]
    }

    mlp = MLPRegressor()
    grid_search = GridSearchCV(estimator=mlp, param_grid=parameters, cv=3, n_jobs=1, verbose=5, scoring='neg_mean_squared_error')

    grid_search.fit(
        np.concatenate((X_train, X_val), axis=0), 
        np.concatenate((y_train, y_val), axis=0)
    )

    print("Best Parameters found:")
    print(grid_search.best_params_)

    best_mlp = grid_search.best_estimator_

    model_path = os.path.join("Models", f"{model_name}.pkl")
    save_model(best_mlp, model_path)
    loaded_mlp = load_model(model_path)

    #results_df = evaluate_model_multi_target(best_mlp, X_test, y_test, dates_test, scaler, model_name='MLP Regressor Predictions')
    #metrics = {"MSE": mse_direct, "MAE": mae_rescaled}

    model_paths = {
        f"{model_name} Archived": os.path.join("Model Archive", f"{model_name}.pkl"),
        f"{model_name}": model_path
    }
    archive_path = os.path.join("Model Archive", f"{model_name}.pkl")
    evaluation_df = evaluate_and_save_best_model(model_paths, X_test, y_test, dates_test, scaler, archive_path, sklearn=True)
    # metrics = {
    #     "MSE": evaluation_df[evaluation_df["Model"]==model_filename]["MSE (scaled)"],
    #     "MAE": evaluation_df[evaluation_df["Model"]==model_filename]["MAE (rescaled)"]
    # }
    update_and_save_metrics(evaluation_df[evaluation_df["Model"] == model_name].copy(), dataset_name, hyperparameters=grid_search.best_params_, excel_dir=".", file_name='training_results.xlsx')
    return mlp, evaluation_df
# Ensembler for univariate Predictors
class MultivariateEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='xgb', model_params=None, outputs=12, model_name="Ensemble",mlp_hyperparameters=MLP_hyperparameters(), dev_save_directory="Models", dataset_name="OECD", archive_save_directory="Model Archive"):
        self.model_type = model_type
        self.mlp_hyperparameters=mlp_hyperparameters
        self.model_name = f"{model_type} {model_name}" if model_name=="Ensemble" else model_name
        self.model_params = model_params if model_params is not None else {}
        self.outputs = outputs
        self.dev_save_directory = dev_save_directory
        self.save_path = os.path.join(self.dev_save_directory, f"{dataset_name} {self.model_name}")
        self.dataset_name = dataset_name
        self.archive_save_directory = archive_save_directory
        self.archive_save_path = os.path.join(self.archive_save_directory, f"{dataset_name} {self.model_name}")
        self.models = self._initialize_models()
    def _initialize_models(self):
        # Initialisierung der Modelle entsprechend dem angegebenen Typ
        if self.model_type == 'xgb':
            return [XGBRegressor(**self.model_params) for _ in range(self.outputs)]
        elif self.model_type == 'linear':
            return [LinearRegression(**self.model_params) for _ in range(self.outputs)]
        elif self.model_type == 'mlp':
            return [MLPRegressor(
                    hidden_layer_sizes=self.mlp_hyperparameters.hidden_layer_sizes,
                    max_iter=self.mlp_hyperparameters.max_iter,
                    learning_rate_init=self.mlp_hyperparameters.learning_rate_init,
                    alpha=self.mlp_hyperparameters.alpha,
                    solver=self.mlp_hyperparameters.solver,
                    validation_fraction=self.mlp_hyperparameters.validation_fraction,
                    n_iter_no_change=self.mlp_hyperparameters.n_iter_no_change
                ) for _ in range(self.outputs)]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    def fit(self, X, y):
        # X ist die Eingabematrix, y ist eine Matrix mit 12 Spalten (eine für jeden Monat)
        for month in range(self.outputs):
            self.models[month].fit(X, y[:, month])
        return self
    def predict(self, X):
        # Vorhersagen für alle 12 Monate generieren
        predictions = np.zeros((X.shape[0], self.outputs))
        for month in range(self.outputs):
            predictions[:, month] = self.models[month].predict(X)
        return predictions
    def model_single_month(self, month=1):
        return self.models[month]
    def explain(self, X, feature_names, month=3, X_index=None, check_values=False):
        model = self.model_single_month(month)
        
        if self.model_type == 'mlp':
            return explain_shap(model, X, feature_names, X_index=X_index, check_values=check_values, summary_output_index=month)

        elif self.model_type == 'linear':
            coefficients = model.coef_
            explanation = {feature_names[i]: coef for i, coef in enumerate(coefficients)}
            # Sortiere die Erklärungen nach den Koeffizienten in absteigender Reihenfolge
            sorted_explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))
            # Wandle die sortierten Erklärungen in einen DataFrame um
            df_explanation = pd.DataFrame(list(sorted_explanation.items()), columns=['Feature', 'Coefficient'])
            # Aggregiere die Koeffizienten für jede Zeitreihe
            explanation_aggregated = defaultdict(float)
            for feature, coef in explanation.items():
                base_name = feature.rsplit('_', 1)[0]  # Entferne den Suffix '_i'
                explanation_aggregated[base_name] += coef
            sorted_explanation_aggregated = dict(sorted(explanation_aggregated.items(), key=lambda item: abs(item[1]), reverse=True))
            # Wandle das aggregierte Dictionary in einen DataFrame um
            df_explanation_aggregated = pd.DataFrame(list(sorted_explanation_aggregated.items()), columns=['Feature', 'Total Coefficient'])
            shap_values = df_explanation_aggregated['Total Importance'].values.reshape(1, -1)
            features = df_explanation_aggregated['Feature'].values
            shap.summary_plot(shap_values, features, plot_type="bar", show=False)
            plt.title("Aggregated Feature Importance (XGBoost Model)")
            plt.show()
            return df_explanation, df_explanation_aggregated

        elif self.model_type == 'xgb':
            importance = model.feature_importances_
            explanation = {feature_names[i]: imp for i, imp in enumerate(importance)}
            # Sortiere die Erklärungen nach den Wichtigkeiten in absteigender Reihenfolge
            sorted_explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))
            # Wandle die sortierten Erklärungen in einen DataFrame um
            df_explanation = pd.DataFrame(list(sorted_explanation.items()), columns=['Feature', 'Importance'])
            # Aggregiere die Wichtigkeiten für jede Zeitreihe
            explanation_aggregated = defaultdict(float)
            for feature, imp in explanation.items():
                base_name = feature.rsplit('_', 1)[0]  # Entferne den Suffix '_i'
                explanation_aggregated[base_name] += imp
            sorted_explanation_aggregated = dict(sorted(explanation_aggregated.items(), key=lambda item: abs(item[1]), reverse=True))
            # Wandle das aggregierte Dictionary in einen DataFrame um
            df_explanation_aggregated = pd.DataFrame(list(sorted_explanation_aggregated.items()), columns=['Feature', 'Total Importance'])
            shap_values = df_explanation_aggregated['Total Importance'].values.reshape(1, -1)
            features = df_explanation_aggregated['Feature'].values
            shap.summary_plot(shap_values, features, plot_type="bar", show=False)
            plt.title("Aggregated Feature Importance (XGBoost Model)")
            plt.show()
            return df_explanation, df_explanation_aggregated

        else:
            raise ValueError(f"Explain method is not implemented for {self.model_type} models.")
    def save(self, path=None):
        path = self.save_path if path == None else path
        if self.model_type == "mlp":
            if not os.path.exists(path):
                os.makedirs(path)
            for i, model in enumerate(self.models):
                model_path = os.path.join(path, f"model_{i}.pkl")
                joblib.dump(model, model_path)
    def load(self, path=None):
        path = self.archive_save_path if path == None else path
        for i in range(self.outputs):
            model_path = os.path.join(path, f"model_{i}.pkl")
            self.models[i] = joblib.load(model_path)
def build_train_evaluate_ensemble(model_type, model_name:str, X_train, y_train, X_test, y_test, dates_test, target_scaler, feature_names=None, dataset_name="OECD", mlp_hyperparameters=MLP_hyperparameters()):
    model = MultivariateEnsemble(model_type, model_name=model_name, mlp_hyperparameters=mlp_hyperparameters, dataset_name=dataset_name)
    model.fit(X_train, y_train)
    model.save()
    
    if (model_type == "mlp"):
        model_paths = {
            f"{model_name} Archived": model.archive_save_path,
            f"{model_name}": model.save_path
        }
        eval_results_df = evaluate_and_save_best_model(model_paths, X_test, y_test, dates_test, target_scaler, model.archive_save_path, sklearn=True, multi=True, mlp_ensemble=True, dataset_name=dataset_name)
    else:
        eval_results_df = evaluate_model_multi_target(model,X_test, y_test, dates_test, target_scaler, model.model_name, plotly=True)
    hyperparameters = model.mlp_hyperparameters.get_dict() if model.model_type == "mlp" else None
    if hyperparameters:
        update_and_save_metrics(eval_results_df[eval_results_df["Model"] == model_name].copy(), dataset_name, hyperparameters=hyperparameters, excel_dir=".", file_name='training_results.xlsx')
    return model, eval_results_df
# Fully Connected Neural Network
def build_fcnn(X_train, layer_sizes, dropout_rates, l2_reg_values, activation_function, output_size, initial_lr, optimizer, loss="mse"):
    if dropout_rates is None:   
        dropout_rates = [0] * len(layer_sizes)
    else:
        if len(dropout_rates) < len(layer_sizes):
            dropout_rates.extend([0] * (len(layer_sizes) - len(dropout_rates)))
        elif len(dropout_rates) > len(layer_sizes):
            dropout_rates = dropout_rates[:len(layer_sizes)]

    if l2_reg_values is None:
        l2_reg_values = [0] * len(layer_sizes)
    else:
        if len(l2_reg_values) < len(layer_sizes):
            l2_reg_values.extend([0] * (len(layer_sizes) - len(l2_reg_values)))
        elif len(l2_reg_values) > len(layer_sizes):
            l2_reg_values = l2_reg_values[:len(layer_sizes)]

    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
    for size, dropout, l2_reg_value in zip(layer_sizes, dropout_rates, l2_reg_values):
        model.add(Dense(size, activation=activation_function, kernel_regularizer=l2(l2_reg_value)))
        #model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(output_size))

    if optimizer == 'adam':
        optimizer_instance = Adam(learning_rate=initial_lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    model.compile(optimizer=optimizer_instance, loss=loss)

    return model
def train_nn(model, X_train, y_train, X_val, y_val, X_test, y_test, dates, scaler, save_path, model_name, batch_size, epochs, initial_lr,
             lr_decay_rate, lr_decay_step, lr_rop_decay_rate, lr_rop_patience, lr_rop_min,
             dynamic_batch_decay_factor, dynamic_batch_decay_epoch, model_title):

    checkpoint = ModelCheckpoint(os.path.join(save_path, f"{model_name}.keras"), save_best_only=True, monitor='val_loss', mode='min')
    lr_scheduler_cb = LearningRateScheduler(lambda epoch, lr: lr_scheduler_function(epoch, lr, lr_decay_rate, lr_decay_step))
    lr_tracker = LearningRateTracker()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_rop_decay_rate, patience=lr_rop_patience, min_lr=lr_rop_min)
    dynamic_batch_size_callback = DynamicBatchSizeCallback(initial_batch_size=batch_size, decay_factor=dynamic_batch_decay_factor, decay_epoch=dynamic_batch_decay_epoch)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint, lr_tracker, lr_scheduler_cb, reduce_lr, dynamic_batch_size_callback])
    
    best_model = keras_load_model(os.path.join(save_path, f"{model_name}.keras"))

    plot_training_history(history, model_title)
    history_to_json(history.history, os.path.join(save_path, f"{model_name}_history.json"))

    model_paths = {
        f"{model_name} Archived": os.path.join("Model Archive", f"{model_name}.keras"),
        f"{model_name}": os.path.join("Models", f"{model_name}.keras")
    }
    archive_path = os.path.join("Model Archive", f"{model_name}.keras")
    evaluation_df = evaluate_and_save_best_model(model_paths, X_test, y_test, dates, scaler, archive_path)

    return best_model, history, evaluation_df
def build_and_train_fcnn_model(X_train, y_train, X_val, y_val, X_test, y_test, dates, scaler,
                             save_path="Models", model_filename="FCNN",
                             model_title="Feed Forward Neural Network",
                             epochs=60, batch_size=8, initial_lr=0.001, 
                             loss="mse",
                             layer_sizes=[64, 32], 
                             dropout_rates=[0.2, 0.2],
                             l2_reg_values=None, 
                             activation_function='relu', 
                             output_size=12, 
                             optimizer='adam',
                             lr_decay_rate=1, 
                             lr_decay_step=10, 
                             lr_rop_decay_rate=0.99,
                             lr_rop_patience=5,
                             lr_rop_min=0.00001,
                             dynamic_batch_decay_factor=1, 
                             dynamic_batch_decay_epoch=15):
    
    model = build_fcnn(X_train, layer_sizes, dropout_rates, l2_reg_values, activation_function, output_size, initial_lr, optimizer)
    
    best_model, history, evaluation_df = train_nn(model, X_train, y_train, X_val, y_val, X_test, y_test, dates, scaler, save_path, model_filename, batch_size, epochs, initial_lr,
                                   lr_decay_rate, lr_decay_step, lr_rop_decay_rate, lr_rop_patience, lr_rop_min,
                                   dynamic_batch_decay_factor, dynamic_batch_decay_epoch, model_title)
    # metrics = {
    #     "MSE": evaluation_df[evaluation_df["Model"]==model_filename]["MSE (scaled)"],
    #     "MAE": evaluation_df[evaluation_df["Model"]==model_filename]["MAE (rescaled)"]
    # }
    return best_model, history, evaluation_df
class FCNN_hyperparameters:
    def __init__(self,
                epochs=200, 
                batch_size=30, 
                initial_lr=0.008, 
                loss = "mse", #"mse"
                layer_sizes=[128, 50, 30], 
                dropout_rates=[0.2, 0.2, 0.1],
                l2_reg_values=[0.01, 0.01, 0.01],
                activation_function='relu', 
                output_size=12, 
                optimizer='adam', 
                lr_decay_rate=0.95, # custom lr_decay off
                lr_decay_step=50,
                lr_rop_decay_rate=0.8,
                lr_rop_min= 1e-4,
                lr_rop_patience=10,
                dynamic_batch_decay_factor=2, 
                dynamic_batch_decay_epoch=20) -> None:
        self.epochs = epochs
        self.batch_size = batch_size 
        self.initial_lr = initial_lr 
        self.loss = loss
        self.layer_sizes = layer_sizes
        self.dropout_rates = dropout_rates
        self.l2_reg_values = l2_reg_values
        self.activation_function = activation_function
        self.output_size = output_size
        self.optimizer = optimizer
        self.lr_decay_rate = lr_decay_rate # custom lr_decay off
        self.lr_decay_step = lr_decay_step
        self.lr_rop_decay_rate = lr_rop_decay_rate
        self.lr_rop_min = lr_rop_min
        self.lr_rop_patience = lr_rop_patience
        self.dynamic_batch_decay_factor = dynamic_batch_decay_factor 
        self.dynamic_batch_decay_epoch = dynamic_batch_decay_epoch

    def get_dict(self):
        return {key: value for key, value in self.__dict__.items()}
def build_train_evaluate_fcnn_model(X_train, y_train, X_val, y_val, X_test, y_test, dates, scaler, hyperparameters = FCNN_hyperparameters(), save_path="Models", dataset_name="OECD", model_title="Fully Connected Neural Network"):
    model, history, evaluation_df = build_and_train_fcnn_model(X_train, y_train, X_val, y_val, X_test, y_test, dates, scaler,
                             save_path, f"{dataset_name}_FCNN", model_title,
                             epochs=hyperparameters.epochs,
                             batch_size=hyperparameters.batch_size,
                             initial_lr=hyperparameters.initial_lr,
                             loss=hyperparameters.loss,
                             layer_sizes=hyperparameters.layer_sizes,
                             dropout_rates=hyperparameters.dropout_rates,
                             l2_reg_values=hyperparameters.l2_reg_values,
                             activation_function=hyperparameters.activation_function,
                             output_size=hyperparameters.output_size,
                             optimizer=hyperparameters.optimizer,
                             lr_decay_rate=hyperparameters.lr_decay_rate,
                             lr_decay_step=hyperparameters.lr_decay_step,
                             lr_rop_decay_rate=hyperparameters.lr_rop_decay_rate,
                             lr_rop_patience=hyperparameters.lr_rop_patience,
                             lr_rop_min=hyperparameters.lr_rop_min,
                             dynamic_batch_decay_factor=hyperparameters.dynamic_batch_decay_factor,
                             dynamic_batch_decay_epoch=hyperparameters.dynamic_batch_decay_epoch )
    update_and_save_metrics(evaluation_df[evaluation_df["Model"] == f"{dataset_name}_FCNN"].copy(), dataset_name, hyperparameters.get_dict())
    return model, history, evaluation_df
# Neural Network using FF, TCN, LSTM and Transformer-Encoder
def build_fc_encoder(input_shape, params):
    inputs = Input(shape=input_shape)
    x = inputs
    for size, activation in zip(params['layer_sizes'], params['activation_functions']):
        x = Dense(size, activation=activation)(x)
    return Model(inputs, x, name='ff_encoder')
@register_keras_serializable()
def split_tensor(input_tensor, num_splits, axis):
    return tf.split(input_tensor, num_splits, axis=axis)
def build_tcn_encoder(input_shape, params):
    inputs = Input(shape=input_shape) # shape is (time series steps, features)

    # Split the input tensor by features into separate time series to define 1D-Convs for each time series
    # input shape of lambda based on inputs: (None/Batchsize, steps, 146)
    split_inputs = Lambda(split_tensor, arguments={'num_splits': input_shape[1], 'axis': 2})(inputs)
    
    conv_outputs = []
    for i in range(input_shape[1]):
        x = split_inputs[i]  # shape is (batch_size, steps, 1)
        for filters, kernel_size in zip(params['tcn_filters'], params['tcn_kernel_sizes']):
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
        x = AveragePooling1D(pool_size=params['tcn_pool_size'])(x)
        conv_outputs.append(Flatten()(x))
    
    # Concatenate the outputs of all time series
    concatenated_output = Concatenate()(conv_outputs)
    
    return Model(inputs, concatenated_output, name='tcn_encoder')
def build_lstm_encoder(input_shape, params):
    inputs = Input(shape=input_shape)  # input shape = (timeseries legnth, features)
    x = inputs
    for units in params['lstm_units']:
        # expects shape (batch_size/None, timesteps, features)
        x = LSTM(units=units, return_sequences=True)(x)
    x = LSTM(units=params['final_lstm_units'])(x)
    return Model(inputs, x, name='lstm_encoder')
def build_transformer_encoder(input_shape, params):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=params['num_heads'], key_dim=params['key_dim'])(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x = Dropout(params['dropout_rate'])(x)
    outputs = Dense(units=params['transformer_units'], activation='relu')(x)
    return Model(inputs, outputs, name='transformer_encoder')
def build_decoder(input_shape, params):
    inputs = Input(shape=input_shape)
    x = inputs
    for size, activation in zip(params['layer_sizes'], params['activation_functions']):
        x = Dense(size, activation=activation)(x)
    outputs = Dense(params['output_size'], activation='linear')(x)
    return Model(inputs, outputs, name='decoder')
def pretrain_ff(input_shape, general_params, encoder_params, checkpoint_path):
    encoder = build_fc_encoder(input_shape, encoder_params)
    encoder_input = Input(shape=input_shape)
    encoded = encoder(encoder_input)
    decoder = build_decoder(encoded.shape[1:], encoder_params)
    decoded = decoder(encoded)
    autoencoder = Model(encoder_input, decoded)
    
    optimizer = Adam(learning_rate=general_params['initial_lr'])
    lr_scheduler_cb = LearningRateScheduler(lambda epoch, lr: lr_scheduler_function(epoch, lr, general_params['lr_decay_rate'], general_params['lr_decay_step']))
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    
    autoencoder.compile(optimizer=optimizer, loss='mse')
    history = autoencoder.fit(general_params['train_data'], general_params['train_data'],
                              epochs=general_params['epochs'],
                              batch_size=general_params['initial_batch_size'],
                              validation_data=(general_params['val_data'], general_params['val_data']),
                              callbacks=[lr_scheduler_cb, checkpoint_cb],
                              verbose=general_params['verbose'])
    return encoder, history
def pretrain_tcn(input_shape, general_params, encoder_params, checkpoint_path):
    encoder = build_tcn_encoder(input_shape, encoder_params)
    encoder_input = Input(shape=input_shape)
    encoded = encoder(encoder_input)
    decoder_output_shape = input_shape[0] * input_shape[1]
    decoder = build_decoder(encoded.shape[1:], encoder_params)
    decoded = decoder(encoded)
    decoded = Reshape(input_shape)(decoded)
    autoencoder = Model(encoder_input, decoded)
    
    optimizer = Adam(learning_rate=general_params['initial_lr'])
    lr_scheduler_cb = LearningRateScheduler(lambda epoch, lr: lr_scheduler_function(epoch, lr, general_params['lr_decay_rate'], general_params['lr_decay_step']))
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    
    autoencoder.compile(optimizer=optimizer, loss='mse')
    history = autoencoder.fit(general_params['train_data'], general_params['train_data'],
                              epochs=general_params['epochs'],
                              batch_size=general_params['initial_batch_size'],
                              validation_data=(general_params['val_data'], general_params['val_data']),
                              callbacks=[lr_scheduler_cb, checkpoint_cb],
                              verbose=general_params['verbose'])
    return encoder, history
def pretrain_lstm(input_shape, general_params, encoder_params, checkpoint_path):
    encoder = build_lstm_encoder(input_shape, encoder_params)
    encoder_input = Input(shape=input_shape)
    encoded = encoder(encoder_input)
    decoder = build_decoder(encoded.shape[1:], encoder_params)
    decoded = decoder(encoded)
    decoded = Reshape(input_shape)(decoded)
    autoencoder = Model(encoder_input, decoded)
    
    optimizer = Adam(learning_rate=general_params['initial_lr'])
    lr_scheduler_cb = LearningRateScheduler(lambda epoch, lr: lr_scheduler_function(epoch, lr, general_params['lr_decay_rate'], general_params['lr_decay_step']))
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    
    autoencoder.compile(optimizer=optimizer, loss='mse')
    history = autoencoder.fit(general_params['train_data'], general_params['train_data'],
                              epochs=general_params['epochs'],
                              batch_size=general_params['initial_batch_size'],
                              validation_data=(general_params['val_data'], general_params['val_data']),
                              callbacks=[lr_scheduler_cb, checkpoint_cb],
                              verbose=general_params['verbose'])
    return encoder, history
def pretrain_transformer(input_shape, general_params, encoder_params, checkpoint_path):
    encoder = build_transformer_encoder(input_shape, encoder_params)
    encoder_input = Input(shape=input_shape)
    encoded = encoder(encoder_input)
    decoder = build_decoder(encoded.shape[1:], encoder_params)
    decoded = decoder(encoded)
    autoencoder = Model(encoder_input, decoded)
    
    optimizer = Adam(learning_rate=general_params['initial_lr'])
    lr_scheduler_cb = LearningRateScheduler(lambda epoch, lr: lr_scheduler_function(epoch, lr, general_params['lr_decay_rate'], general_params['lr_decay_step']))
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    
    autoencoder.compile(optimizer=optimizer, loss='mse')
    history = autoencoder.fit(general_params['train_data'], general_params['train_data'],
                              epochs=general_params['epochs'],
                              batch_size=general_params['initial_batch_size'],
                              validation_data=(general_params['val_data'], general_params['val_data']),
                              callbacks=[lr_scheduler_cb, checkpoint_cb],
                              verbose=general_params['verbose'])
    return autoencoder, encoder, history
def train_model(model, general_params):
    lr_scheduler_cb = LearningRateScheduler(lambda epoch, lr: lr_scheduler_function(epoch, lr, general_params['lr_decay_rate'], general_params['lr_decay_step']))
    history = model.fit(general_params['train_data'], general_params['train_labels'],
                        epochs=general_params['epochs'],
                        batch_size=general_params['initial_batch_size'],
                        validation_data=(general_params['val_data'], general_params['val_labels']) if ('val_data' in general_params and 'val_labels' in general_params and general_params['val_data'] is not None and general_params['val_labels'] is not None) else None,
                        callbacks=[lr_scheduler_cb],
                        verbose=general_params['verbose'])
    return model, history
def build_nn(input_shape, general_params, encoder_params_list, decoder_params):
    # Build encoders and concatenate their outputs
    inputs = Input(shape=input_shape)
    encoder_outputs = []
    for encoder_params in encoder_params_list:
        if encoder_params['type'] == 'ff':
            encoder = build_fc_encoder(input_shape, encoder_params)
        elif encoder_params['type'] == 'tcn':
            encoder = build_tcn_encoder(input_shape, encoder_params)
        elif encoder_params['type'] == 'lstm':
            encoder = build_lstm_encoder(input_shape, encoder_params)
        elif encoder_params['type'] == 'transformer':
            encoder = build_transformer_encoder(input_shape, encoder_params)
        else:
            encoder = None
        
        if encoder is not None:
            if encoder_params['pretrained']:
                encoder.load_weights(encoder_params['pretrained_model_path'])
                if not encoder_params['finetune']:
                    for layer in encoder.layers:
                        layer.trainable = False
            encoder_output = encoder(inputs)
            if len(encoder_output.shape) > 2:
                encoder_output = Flatten()(encoder_output)
            encoder_outputs.append(encoder_output)
    
    if len(encoder_outputs) > 1:
        concatenated_output = Concatenate()(encoder_outputs)
    else:
        concatenated_output = encoder_outputs[0]

    # Build decoder
    decoder = build_decoder(concatenated_output.shape[1:], decoder_params)
    outputs = decoder(concatenated_output)
    model = Model(inputs, outputs)

    # Select optimizer
    if general_params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=general_params['initial_lr'])
    elif general_params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=general_params['initial_lr'])
    else:
        raise ValueError("Unsupported optimizer type")

    model.compile(optimizer=optimizer, loss=general_params['loss'], metrics=general_params['metrics'])
    return model
class NN_encoder_parameters:
    def __init__(self,
                type,  # 'ff', 'tcn', 'lstm', 'transformer', 'none'
                pretrained= False,
                pretrained_model_path= "",  # Path to pretrained model weights
                finetune= True,
                # FC
                layer_sizes= [128, 32],
                activation_functions= ['relu', 'relu'],
                # TCN
                tcn_filters= [5, 5],
                tcn_kernel_sizes= [6, 2],
                tcn_pool_size= 1,
                # LSTM
                lstm_units= [20],
                final_lstm_units= 20,
                # Transformer
                num_heads= 3,
                key_dim= 16,
                dropout_rate= 0.1,
                transformer_units= 32
                ):
        self.type=type
        self.pretrained= pretrained
        self.pretrained_model_path= pretrained_model_path
        self.finetune= finetune
        # FC
        self.layer_sizes= layer_sizes
        self.activation_functions= activation_functions
        # TCN
        self.tcn_filters= tcn_filters
        self.tcn_kernel_sizes= tcn_kernel_sizes
        self.tcn_pool_size= tcn_pool_size
        # LSTM
        self.lstm_units = lstm_units
        self.final_lstm_units= final_lstm_units
        # Transformer
        self.num_heads=num_heads
        self.key_dim= key_dim
        self.dropout_rate= dropout_rate
        self.transformer_units= transformer_units
    
    def get_dict(self):
        return {key: value for key, value in self.__dict__.items()}
class NN_decoder_parameters:
    def __init__(self,
                layer_sizes= [64, 32],
                activation_functions= ['relu', 'relu'],
                output_size= 12):
        self.layer_sizes= layer_sizes
        self.activation_functions= activation_functions
        self.output_size= output_size
    def get_dict(self):
        return {key: value for key, value in self.__dict__.items()}
class NN_hyperparameters:
    def __init__(self, 
                epochs = 100,
                initial_batch_size = 33,
                batch_size_decay_factor = 0.5,
                batch_size_decay_step = 10,
                initial_lr = 0.01,
                lr_decay_rate = 0.9,
                lr_decay_step = 20,
                optimizer = 'adam',
                loss = 'mse',
                metrics = ['mae'],
                verbose = 1,
                model_save_path = "Models",
                model_filename = f"NN",
                model_title = "Neural Network with FF, TCN, LSTM and Transformer Encoders",
                lr_rop_decay_rate = 0.8,
                lr_rop_patience = 10,
                lr_rop_min = 1e-4,
                dynamic_batch_decay_factor = 1.5,
                dynamic_batch_decay_epoch = 10,
                encoder_params_list: list[NN_encoder_parameters] = [NN_encoder_parameters(type='ff'), NN_encoder_parameters(type='tcn'), NN_encoder_parameters(type='lstm'), NN_encoder_parameters('transformer')],
                decoder_params = NN_decoder_parameters() ):
        self.general_params = {
            'epochs': epochs,
            'initial_batch_size': initial_batch_size,
            'batch_size_decay_factor': batch_size_decay_factor,
            'batch_size_decay_step': batch_size_decay_step,
            'initial_lr': initial_lr,
            'lr_decay_rate': lr_decay_rate,
            'lr_decay_step': lr_decay_step,
            'optimizer': optimizer,
            'loss': loss,
            'metrics': metrics,
            'verbose': verbose,
            'model_save_path': model_save_path,
            'model_filename': model_filename,
            'model_title': model_title,
            'lr_rop_decay_rate': lr_rop_decay_rate,
            'lr_rop_patience': lr_rop_patience,
            'lr_rop_min': lr_rop_min,
            'dynamic_batch_decay_factor': dynamic_batch_decay_factor,
            'dynamic_batch_decay_epoch': dynamic_batch_decay_epoch
        }

        self.encoder_params_list = []
        for enc in encoder_params_list:
            self.encoder_params_list.append(enc.get_dict())
        
        self.decoder_params = decoder_params.get_dict()

    def get_dict(self):
        return {key: value for key, value in self.__dict__.items()}
    def get_dict_flat(self):
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                new_key = new_key.replace('general_params_', '')
                new_key = new_key.replace('encoder_params_list', 'encoder')
                new_key = new_key.replace('decoder_params', 'decoder')
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            items.extend(flatten_dict(item, f"{new_key}_{i}").items())
                        else:
                            items.append((f"{new_key}_{i}", item))
                else:
                    items.append((new_key, v))
            return dict(items)
        return flatten_dict(self.__dict__)
def build_train_evaluate_nn(X_train, y_train, X_val, y_val, X_test, y_test, dates_test, target_scaler, model_save_path = "Models", dataset_name = f"OECD", model_title = "Neural Network with FF, TCN, LSTM and Transformer Encoders", hyperparameters=NN_hyperparameters()):    
    input_shape = X_train.shape[1:]
    model = build_nn(input_shape, hyperparameters.general_params, hyperparameters.encoder_params_list, hyperparameters.decoder_params)
    model.summary()
    model, history, evaluation_df= train_nn(    #train_model(model, general_params)
        model, 
        X_train, y_train, X_val, y_val, X_test, y_test, dates_test, target_scaler,
        save_path=model_save_path, model_name=f"{dataset_name}_NN", 
        batch_size=hyperparameters.general_params['initial_batch_size'], epochs=hyperparameters.general_params['epochs'], 
        initial_lr=hyperparameters.general_params['initial_lr'], lr_decay_rate=hyperparameters.general_params['lr_decay_rate'], lr_decay_step=hyperparameters.general_params['lr_decay_step'], 
        lr_rop_decay_rate=hyperparameters.general_params['lr_rop_decay_rate'], lr_rop_patience=hyperparameters.general_params['lr_rop_patience'], lr_rop_min=hyperparameters.general_params['lr_rop_min'],
        dynamic_batch_decay_factor=hyperparameters.general_params['dynamic_batch_decay_factor'], dynamic_batch_decay_epoch=hyperparameters.general_params['dynamic_batch_decay_epoch'], 
        model_title=model_title)
    update_and_save_metrics(evaluation_df[evaluation_df["Model"] == f"{dataset_name}_NN"].copy(), dataset_name, hyperparameters.get_dict_flat())
    return model, history, evaluation_df

def plot_training_history(history, model_name):
    if history is not None:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plotting Training and Validation Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{model_name} Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')
        
        # Creating a second y-axis for the learning rate
        ax2 = ax1.twinx()
        if 'lr' in history.history:
            ax2.plot(history.history['lr'], 'r--', label='Learning Rate')
            ax2.set_ylabel('Learning Rate')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
def plot_training_history(history, model_name):
    if history is not None:
        fig = go.Figure()

        # Adding title and labels
        fig.update_layout(
            title=f'{model_name} Training and Validation Loss',
            xaxis_title='Epochs',
            yaxis_title='Loss'
        )
        
        # Plotting Training and Validation Loss
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            mode='lines',
            name='Training Loss'
        ))
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            mode='lines',
            name='Validation Loss'
        ))
        if "batch_size" in history.history:
            fig.add_trace(go.Scatter(
                y=history.history['batch_size'],
                mode='lines',
                name='Batch Size',
                yaxis='y2'
            ))
            # Add second y-axis to the figure
            fig.update_layout(
                yaxis2=dict(
                    title='Learning Rate',
                    overlaying='y',
                    side='right'
                )
            )

        if 'lr' in history.history:
            fig.add_trace(go.Scatter(
                y=history.history['lr'],
                mode='lines',
                name='Learning Rate'
            ))
            
        
        fig.show()


# EVALUATION 
def evaluate_and_save_best_model(model_paths, X, y, dates, target_scaler, archive_path, sklearn=False, multi=True, mlp_ensemble=False, dataset_name="OECD"):
    models={}

    def save_ensemble():
        if os.path.exists(best_model_path):
            model = MultivariateEnsemble(model_type="mlp", dev_save_directory=best_model_path, model_name="MLP Ensemble", dataset_name=dataset_name)
            model.load(path=best_model_path)
            model.save(path=archive_path)
        else:
            print(f"Modelldatei '{best_model_path}' nicht gefunden.")

    def save_model_to_archive():
        if mlp_ensemble:
            save_ensemble()
        else:
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, archive_path)
                print(f"Modelldatei '{best_model_path}' wurde nach '{archive_path}' kopiert.")
            else:
                print(f"Modelldatei '{best_model_path}' nicht gefunden.")
        if not sklearn and not mlp_ensemble:
            history_file = best_model_path.replace(".keras", "_history.json")
            archive_history_file = archive_path.replace(".keras", "_history.json")
            if os.path.exists(history_file):
                shutil.copy(history_file, archive_history_file)
                print(f"History-Datei '{history_file}' wurde nach '{archive_history_file}' kopiert.")
            else:
                print(f"History-Datei '{history_file}' nicht gefunden.")

    if not os.path.exists(archive_path):
        for key in model_paths:
            if "Archive" not in key:
                best_model_path = model_paths[key]
                break
        save_model_to_archive()

    if mlp_ensemble:
        for model_name in model_paths.keys():
            model = MultivariateEnsemble(model_type="mlp", dev_save_directory=model_paths[model_name], model_name="MLP Ensemble", dataset_name=dataset_name)
            model.load(path=model_paths[model_name])
            models[model_name] = model
    elif sklearn:
        for model_name in model_paths.keys():
            models[model_name] = load_model(model_paths[model_name])
    else:
        for model_name in model_paths.keys():
            models[model_name] = keras_load_model(model_paths[model_name])
            
    results_df, best_model_name = compare_evaluations(models, X, y, dates, target_scaler, multi)

    best_model_path = model_paths[best_model_name]
    if best_model_path and best_model_path != archive_path:
        save_model_to_archive()
        
    return results_df
def evaluate_model_single_target(model, X, y, dates, target_scaler, model_name='Model', feature_names=None): # DEPRECATED
    y_pred = model.predict(X)
    y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = target_scaler.inverse_transform(y.reshape(-1, 1))
    y_pred_rescaled_df = pd.DataFrame(y_pred_rescaled, index=dates, columns=["Prediction"])
    y_combined = pd.DataFrame(y_test_rescaled, index=dates, columns=["Ground Truth"]).join(y_pred_rescaled_df, how='inner')
    
    mse_direct = mean_squared_error(y, y_pred)
    mae_rescaled = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled) * 100  # Convert to percentage
    explained_variance = explained_variance_score(y_test_rescaled, y_pred_rescaled) * 100
    results_df = pd.DataFrame({
        'Model': [f"{model_name}"],
        'MSE (scaled)': [mse_direct],
        'MAE (M€)': [mae_rescaled / 1e6],
        # Standard deviation
        # confidence interval
        'R² Score (%)': [r2],
        'Explained Variance (%)': [explained_variance]
    })
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_combined["Ground Truth"], label='True Values')
    plt.plot(y_combined["Prediction"], label='Predicted Values')
    plt.title(f'{model_name} Prediction')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compute residuals for histogram
    residuals = y_test_rescaled - y_pred_rescaled
    residuals_flat = residuals.flatten()

    # Calculate quantiles for 50% and 80%
    q25, q75 = np.percentile(residuals_flat, [25, 75])
    q10, q90 = np.percentile(residuals_flat, [10, 90])
    range_50 = q75 - q25
    range_80 = q90 - q10

    # Plot the histogram of prediction errors
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_flat, bins=50, edgecolor='k', alpha=0.7)

    # Add color bands for quantiles
    plt.axvspan(q25, q75, color='lightcoral', alpha=0.5, label='50% Error Range')
    plt.axvspan(q10, q90, color='indianred', alpha=0.3, label='80% Error Range')

    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Residual plot
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test_rescaled.flatten(), residuals_flat, alpha=0.5)
    # plt.axhline(0, color='red', linestyle='--')
    # plt.xlabel('True Values')
    # plt.ylabel('Residuals')
    # plt.title('Residuals vs True Values')
    # plt.show()

    # Binned residual plot
    df = pd.DataFrame({'True': y_test_rescaled.flatten(), 'Pred': y_pred_rescaled.flatten()})
    df['Error'] = df['True'] - df['Pred']
    df['Bin'] = pd.qcut(df['True'], q=10, duplicates='drop')
    binned_residuals = df.groupby('Bin').apply(lambda x: x['Error'].mean())

    binned_residuals.plot(kind='bar', color='skyblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Bins of True Values')
    plt.ylabel('Mean Residuals')
    plt.title('Binned Residuals')
    plt.show()

    # SHAP analysis for model explanation
    if feature_names is not None:
        explain_shap(model, X, feature_names)

    # Output significant statements based on metrics
    print("\nModel Evaluation Summary:")
    print(f"Model: {model_name}")
    print(f"Genauigkeit: Die durchschnittliche absolute Abweichung beträgt {mae_rescaled*1e-6:.3f} M€.")
    print(f"Sensitivität: Das Modell erklärt {r2:.2f}% der Gesamtvarianz der Zielvariable.")
    print(f"Sensitivität: Das Modell reagiert effektiv auf {explained_variance:.4f}% der signifikanten Variationen der Zielvariable.")
    print(f"Präzision: 50 % der Vorhersagen weisen einen Fehler im Bereich von {q25*1e-6:.4f}M€ bis {q75*1e-6:.3f}M€")
    print(f"Präzision: 80 % der Vorhersagen weisen einen Fehler im Bereich von {q10*1e-6:.4f}M€ bis {q90*1e-6:.3f}M€")


    return results_df
def evaluate_model_multi_target(model, X, y, dates, target_scaler, model_name='Model', feature_names=None, plotly=False, show=True, return_plot=False):
    y_pred = model.predict(X)
    y_pred_rescaled = target_scaler.inverse_transform(y_pred)
    y_test_rescaled = target_scaler.inverse_transform(y)


    mse_direct = mean_squared_error(y, y_pred)
    mae_rescaled = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled,multioutput='variance_weighted') * 100  # Convert to percentage
    explained_variance = explained_variance_score(y_test_rescaled, y_pred_rescaled) * 100
    results_df = pd.DataFrame({
        'Model': [f"{model_name}"],
        'MSE (scaled)': [mse_direct],
        'MAE (M€)': [mae_rescaled / 1e6],
        # Standard deviation
        # confidence interval
        'R² Score (%)': [r2],
        'Explained Variance (%)': [explained_variance]
    })

    if plotly:
        # Initialize the DataFrame with the Ground Truth for the first month
        data = {'Ground Truth': y_test_rescaled[:, 0]}
        shifted_dates = dates.copy()

        # Build the DataFrame with shifted predictions
        for i in range(12):
            predictions_col_name = f'Prediction Month {i+1}'
            if i == 0:
                data[predictions_col_name] = y_pred_rescaled[:, i]
            else:
                # Shift the predictions forward by one month
                shifted_prediction = np.append([np.nan] * i, y_pred_rescaled[:, i])
                for key, value in data.items():
                    if key != "Ground Truth":
                        data[key] = np.append(data[key], [np.nan])
                data[predictions_col_name] = shifted_prediction
                # Extend the Ground Truth with the last value of each month
                data['Ground Truth'] = np.append(data['Ground Truth'], y_test_rescaled[-1, i])

                # Extend dates similarly
                shifted_dates = np.append(shifted_dates, dates[-1] + pd.DateOffset(months=i))

        # Create the DataFrame
        y_combined = pd.DataFrame(data, index=shifted_dates[:len(data['Ground Truth'])])
        y_combined = y_combined.rename_axis("Date")

        def create_color_palette(greens_needed=12, main_plots=1):
            colors = color_palette[:main_plots]
            colors.insert(0, color_palette[0])
            # Füge Grüntöne für die restlichen Spalten hinzu
            greens = plt.get_cmap('Greens', 2*greens_needed)
            for i in range(greens_needed):
                rgba = greens((i + 0.7*greens_needed) / (1.8*greens_needed))
                rgb = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
                colors.append(rgb)
            return colors

        # Use create_plot function once
        plot, img = create_plot(
            df=y_combined.reset_index(),
            highlight_columns=['Ground Truth'] + [f'Prediction Month {i+1}' for i in range(12)],
            chart_title=f'{model_name} Prediction',
            xlabel='Samples',
            ylabel='Values',
            color_palette= create_color_palette(),
            default_columns=["Ground Truth"]
        )
        if show:
            plot.show()

    else:
        fig, axs = plt.subplots(12, 1, figsize=(14, 28), sharex=True)
        for i in range(12):
            y_pred_rescaled_df = pd.DataFrame(y_pred_rescaled[:, i], index=dates, columns=[f"Prediction Month {i+1}"])
            y_combined = pd.DataFrame(y_test_rescaled[:, i], index=dates, columns=[f"Ground Truth Month {i+1}"]).join(y_pred_rescaled_df, how='inner')
            
            axs[i].plot(y_combined[f"Ground Truth Month {i+1}"], label='True Values')
            axs[i].plot(y_combined[f"Prediction Month {i+1}"], label='Predicted Values')
            axs[i].set_title(f'Month {i+1}')
            axs[i].legend()

        plt.suptitle(f'{model_name} Prediction')
        plt.xlabel('Samples')
        plt.tight_layout()
        if show:
            plt.show()

    

    # Compute residuals for histogram
    residuals = y_test_rescaled - y_pred_rescaled
    residuals_flat = residuals.flatten()

    # Calculate quantiles for 50% and 80%
    q25, q75 = np.percentile(residuals_flat, [25, 75])
    q10, q90 = np.percentile(residuals_flat, [10, 90])
    range_50 = q75 - q25
    range_80 = q90 - q10

    # Plot the histogram of prediction errors
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_flat, bins=50, edgecolor='k', alpha=0.7)

    # Add color bands for quantiles
    plt.axvspan(q25, q75, color='lightcoral', alpha=0.5, label='50% Error Range')
    plt.axvspan(q10, q90, color='indianred', alpha=0.3, label='80% Error Range')

    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()

    # Residual plot
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test_rescaled.flatten(), residuals_flat, alpha=0.5)
    # plt.axhline(0, color='red', linestyle='--')
    # plt.xlabel('True Values')
    # plt.ylabel('Residuals')
    # plt.title('Residuals vs True Values')
    # plt.show()

    # Binned residual plot
    df = pd.DataFrame({'True': y_test_rescaled.flatten(), 'Pred': y_pred_rescaled.flatten()})
    df['Error'] = df['True'] - df['Pred']
    df['Bin'] = pd.qcut(df['True'], q=10, duplicates='drop')
    binned_residuals = df.groupby('Bin').apply(lambda x: x['Error'].mean())

    binned_residuals.plot(kind='bar', color='skyblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Bins of True Values')
    plt.ylabel('Mean Residuals')
    plt.title('Binned Residuals')
    if show:
        plt.show()

    # SHAP analysis for model explanation
    if feature_names is not None:
        if isinstance(model, MultivariateEnsemble):
            model.explain(X, feature_names=feature_names, month=3)
        else:
            explain_shap(model, X, feature_names)

    # Output significant statements based on metrics
    print("\nModel Evaluation Summary:")
    print(f"Model: {model_name}")
    print(f"Genauigkeit: Die durchschnittliche absolute Abweichung beträgt {mae_rescaled*1e-6:.3f} M€.")
    print(f"Sensitivität: Das Modell erklärt {r2:.2f}% der Gesamtvarianz der Zielvariable.")
    print(f"Sensitivität: Das Modell reagiert effektiv auf {explained_variance:.4f}% der signifikanten Variationen der Zielvariable.")
    print(f"Präzision: 50 % der Vorhersagen weisen einen Fehler im Bereich von {q25*1e-6:.4f}M€ bis {q75*1e-6:.3f}M€")
    print(f"Präzision: 80 % der Vorhersagen weisen einen Fehler im Bereich von {q10*1e-6:.4f}M€ bis {q90*1e-6:.3f}M€")

    if return_plot:
        return results_df, plot
    return results_df
def compare_evaluations(models, X, y, dates, target_scaler, multi=True):
    results_dfs = []
    for name, model in models.items():
        if multi:
            ev_df = evaluate_model_multi_target(model, X, y, dates, target_scaler, name)
        else:
            ev_df = evaluate_model_single_target(model, X, y, dates, target_scaler, name)
        results_dfs.append(ev_df)
    results_df = pd.concat(results_dfs, ignore_index=True)
    best_model_row = results_df.loc[results_df['MAE (M€)'].idxmin()]
    best_model_name = best_model_row['Model']
    return results_df, best_model_name


# PREDICTION
def prediction_report(dataset_name, dataset:Dataset, filename="sales_predictions_report.xlsx"):
    # Create predictions data
    predictions_df = future_predictions(prediction_models(dataset_name), dataset)
    save_results_to_excel({dataset_name: predictions_df}, filename)

    # Create predictions plot
    pred_plot, pred_plot_imgstream = predictions_plot(dataset.target_df, predictions_df)

    predictions_df.set_index(["Prediction Baseline", "Date"], inplace=True)

    return predictions_df, pred_plot, pred_plot_imgstream

def prediction_models(dataset_name):
    model_paths = {
            "FCNN": os.path.join("Model Archive", f"{dataset_name}_FCNN.keras"),
            "NN": os.path.join("Model Archive", f"{dataset_name}_NN.keras")
        }
    prediction_models = {model: keras_load_model(model_paths[model]) for model in model_paths.keys()}
    return prediction_models
def future_predictions(models, data:Dataset):
    def prediction(model, target_scaler, input):
        return target_scaler.inverse_transform(model.predict(input)).tolist()[0]
    
    input = data.X_multi_all[len(data.X_multi_all)-1]
    date = data.dates_multi_all[len(data.dates_multi_all)-1]

    prediction_data = { "Prediction Baseline": [date for i in range(1,13)],
                        "Date": [date + relativedelta(months=i) for i in range(1,13)]}
    
    for name, model in models.items():
        prediction_data[f"Sales Prediction ({name})"] = prediction(model, data.target_scaler, np.array([input]))

    predictions = pd.DataFrame(data=prediction_data)
    return predictions
def future_predictions_multiple_datasets(datasets_dict:dict[str,Dataset], filename="sales_predictions.xlsx"):
    predictions_dict = {}
    for name, data in datasets_dict.items():
        predictions_dict[name] = future_predictions(prediction_models(name), name, Dataset(data))

    def add_dataset_col(df, dataset):
        df["Dataset"] = dataset
        return df

    predictions_dfs = [add_dataset_col(predictions_df.copy(), dataset) for dataset, predictions_df in predictions_dict.items()]

    predictions_df = pd.concat(predictions_dfs, ignore_index=True)
    predictions_df = predictions_df[["Dataset"]]

    save_results_to_excel({"All": predictions_df} | predictions_dict, filename)

    return predictions_df
def predictions_plot(historics_df, predictions_df):
    historics_df = historics_df[["T"]]
    historics_df.rename(columns={"T": "Order Intake"}, inplace=True)
    predictions_df = pd.DataFrame(predictions_df[predictions_df.columns.drop("Prediction Baseline")])
    predictions_df.set_index("Date", inplace=True)
    predictions_df = pd.concat([predictions_df, historics_df], axis=0)
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={"index": "Date"}, inplace=True)
    predictions_df.set_index("Date", inplace=True)

    oi_pred_plot, oi_pred_plot_imgstream = create_plot(
        df=predictions_df.reset_index(), 
        highlight_columns=[], 
        color_palette=color_palette,
        chart_title='Order Intake Predictions',
        xlabel='Datum',
        ylabel='Wert (in M€)',
        scale_factor=1e-6
    )
    return oi_pred_plot, oi_pred_plot_imgstream


# EXPLANATION
def explain_model(model, dataset:Dataset, summary_output_index=3):
    explain_shap(model, dataset.X_test_multi, dataset.time_series, summary_output_index=summary_output_index)
def explain_latest_prediction(model, dataset:Dataset, summary_output_index=3, top_n=20, extend_features=False):
    feature_names = extended_features(dataset.time_series) if extend_features else dataset.time_series
    X = dataset.X_multi_all[len(dataset.X_multi_all)-1:]
    explain_shap_for_instance(model, dataset.X_test_multi, X, feature_names=feature_names, summary_output_index=summary_output_index, extended_features=extend_features, top_n=top_n)

def explain_shap(model, X, feature_names, X_index=None, check_values=False, summary_output_index=3):
    shap_values = None
    explainer = None
    if isinstance(model, (keras.models.Model, keras.models.Sequential)):
        shap.explainers._deep.deep_tf.op_handlers["TensorListStack"] = shap.explainers._deep.deep_tf.passthrough 
        shap.explainers._deep.deep_tf.op_handlers["While"] = shap.explainers._deep.deep_tf.passthrough  
        shap.explainers._deep.deep_tf.op_handlers["BatchMatMulV2"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["TensorListFromTensor"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["Neg"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["Split"] = shap.explainers._deep.deep_tf.passthrough

        explainer = shap.DeepExplainer(model, X)
        shap_values = explainer(X).values
    else:
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)
    shap.initjs()

    # Samples auswählen
    # Wähle das erste Element aus `X_test_multi`
    if X_index is not None:
        X_test_sample = X[X_index:X_index+1]  # Form (1, 12, 146)
    else:
        X_test_sample = X

    # Berechne die Vorhersage des Modells
    model_prediction = model.predict(X_test_sample)

    # Prüfung der shap-Werte
    if check_values and X_index is not None:
        # Berechne die Summe der SHAP-Werte für das erste Element
        # Da `shap_values` die Form (11, 12, 146, 12) bzw (n_samples, n_timesteps, n_features, n_outputs) hat, summieren wir über die Feature-Dimensionen.
        shap_sums = np.sum(shap_values[X_index], axis=(0, 1))  # Form (12,)

        # Überprüfe den erwarteten Wert (Bias) des Erklärers
        expected_values = explainer.expected_value

        # Vergleiche für jeden Output
        for output_index in range(len(shap_sums)):
            prediction = model_prediction[0][output_index]
            shap_sum_plus_bias = expected_values[output_index] + shap_sums[output_index]
            print(f"Output {output_index}:")
            print(f"Model Prediction: {prediction}")
            print(f"SHAP Sum + Expected Value: {shap_sum_plus_bias}")
            print(f"Difference: {abs(prediction - shap_sum_plus_bias)}")

    # Umformung der shap-Werte für den Summary Plot
    shap_values_2D = shap_values[:, :, :, summary_output_index].reshape(-1, shap_values.shape[2])
    X_test_2D = X.reshape(-1, X.shape[2])
    shap_values_2D.shape, X_test_2D.shape, len(feature_names)
    x_test_2d = pd.DataFrame(data=X_test_2D, columns = feature_names)

    shap.summary_plot(shap_values_2D, x_test_2d, max_display=30)
def explain_shap_for_instance(model, X_all, X, feature_names, summary_output_index=3, top_n=20, extended_features=False):
    # Initialisiere den Explainer und berechne die SHAP-Werte
    explainer = None
    shap_values = None
    if isinstance(model, (keras.models.Model, keras.models.Sequential)):
        # Eventuell sind die TensorFlow-Passthroughs nötig
        shap.explainers._deep.deep_tf.op_handlers["TensorListStack"] = shap.explainers._deep.deep_tf.passthrough 
        shap.explainers._deep.deep_tf.op_handlers["While"] = shap.explainers._deep.deep_tf.passthrough  
        shap.explainers._deep.deep_tf.op_handlers["BatchMatMulV2"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["TensorListFromTensor"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["Neg"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["Split"] = shap.explainers._deep.deep_tf.passthrough

        explainer = shap.DeepExplainer(model, X_all)
        shap_values = explainer(X).values
    else:
        explainer = shap.KernelExplainer(model.predict, X_all)
        shap_values = explainer.shap_values(X)

    # Extrahiere SHAP-Werte für den bestimmten Output-Index
    shap_values_instance = shap_values[:, :, :, summary_output_index]
    expected_value = explainer.expected_value[summary_output_index].numpy()


    def reduce_shap_values(shap_values, X, feature_names):
        shap_values_abs = np.abs(shap_values.flatten())
        sorted_indices = np.argsort(shap_values_abs)
        top_indices = sorted_indices[-top_n:]
        remaining_indices = sorted_indices[:-top_n]
        combined_shap_value = np.sum(shap_values.flatten()[remaining_indices])
        shap_values_top = np.concatenate((shap_values.flatten()[top_indices], [combined_shap_value]))
        X_top = np.concatenate((X.flatten()[top_indices], [0]))
        feature_names_top = np.concatenate((np.array(feature_names)[top_indices], ['Combined Others']))
        return shap_values_top, X_top, feature_names_top

    if not extended_features:
        shap_values_instance = np.sum(shap_values_instance, axis=1)
    s, x, f = reduce_shap_values(shap_values_instance, X, feature_names)
    #s, x, f = shap_values_instance, X, feature_names

    # Erstelle einen Force-Plot für die gefilterten Features
    shap.initjs()
    shap.force_plot(expected_value, s.flatten(), x.flatten(), feature_names=f)
    shap.bar_plot(s.flatten(),feature_names=f.flatten(), max_display=top_n+1)


# DOCUMENTATION
def save_results_to_excel(dfs:dict[str,pd.DataFrame], filename):
    with pd.ExcelWriter(filename) as writer:
        for sheetname, df in dfs.items():
            df.to_excel(writer, sheet_name=sheetname, index=False)
def update_and_save_metrics(evaluation_df, dataset_name, hyperparameters, excel_dir=".", file_name='training_results.xlsx'):
    # Erstelle das Verzeichnis, falls es nicht existiert
    print(evaluation_df)
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)

    # Pfad zur Excel-Datei
    file_path = os.path.join(excel_dir, file_name)

    # Füge die 'Dataset' und 'Hyperparameter'-Spalten hinzu
    evaluation_df["Dataset"] = dataset_name
    for param, value in hyperparameters.items():
        evaluation_df[param] = [value]

    # Prüfe, ob die Excel-Datei bereits existiert
    if os.path.exists(file_path):
        # Lese die bestehende Excel-Datei
        existing_results_df = pd.read_excel(file_path)

        # Füge eine numerische Indexspalte hinzu, um Trainingsläufe zu unterscheiden
        if 'Run Index' not in existing_results_df.columns:
            existing_results_df['Run Index'] = 0

        # Setze den Index für den neuen Lauf
        if not existing_results_df.empty:
            max_index = existing_results_df['Run Index'].max()
        else:
            max_index = -1
        evaluation_df['Run Index'] = max_index + 1

        # Füge die neuen Ergebnisse hinzu
        combined_results_df = pd.concat([existing_results_df, evaluation_df], ignore_index=True)
    else:
        # Wenn die Datei nicht existiert, starte den Index bei 0
        evaluation_df['Run Index'] = 0
        combined_results_df = evaluation_df

    # Verschiebe die 'Run Index' Spalte nach 'Model'
    if 'Model' in combined_results_df.columns and 'Run Index' in combined_results_df.columns:
        columns = list(combined_results_df.columns)
        model_index = columns.index('Model')
        run_index_position = model_index + 1  # Nach 'Model'
        # Entferne 'Run Index' und füge es an der gewünschten Position wieder ein
        columns.remove('Run Index')
        columns.insert(run_index_position, 'Run Index')
        combined_results_df = combined_results_df[columns]    

    # Speichere die kombinierten Ergebnisse in der Excel-Datei
    combined_results_df.to_excel(file_path, index=False)