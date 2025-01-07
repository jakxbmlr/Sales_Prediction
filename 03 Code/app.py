from dash import Dash, html, dcc
from data_analysis import *
from data_integration import feature_descriptions, feature_name
from modeling import *

app = Dash()


#pio.renderers.default = "vscode"

oecd_df = pd.read_csv("../04 Data/OECD/OECD_integrated.csv")
oecd_df.set_index("TIME_PERIOD", verify_integrity=True, inplace=True, drop=True)
oecd_df.drop(columns=["Unnamed: 0"], inplace=True)
oecd_df.index = pd.to_datetime(oecd_df.index)
oecd_df.info()

oecd_features = pd.read_csv("../04 Data/OECD/OECD_features.csv")
oecd_features.drop(columns=["Unnamed: 0"], inplace=True)

hg_oi_df = pd.read_csv("../04 Data/HOMAG/OrderIntake_series_monthly.csv")
hg_oi_df.set_index("TIME_PERIOD", verify_integrity=True, inplace=True, drop=True)
hg_oi_df.index = pd.to_datetime(hg_oi_df.index)
hg_oi_df.info()

hg_oi_10y_df = pd.read_csv("../04 Data/HOMAG/OrderIntake_series_monthly_10y.csv")
hg_oi_10y_df.set_index("TIME_PERIOD", verify_integrity=True, inplace=True, drop=True)
hg_oi_10y_df.index = pd.to_datetime(hg_oi_10y_df.index)
hg_oi_10y_df.info()


datasets = {}
datasets['OECD'] = load_preprocess_form_dataset(feature_files = ["../04 Data/OECD/OECD_final.csv"], target_file = "../04 Data/HOMAG/OrderIntake_series_monthly.csv")

current_dataset = "OECD"

(hg_oi_df, oecd_df, # Imported raw data
time_series, target_single, features, targets_multi, # name of target and feature attributes/columns

# Scaler
feature_scaler,
target_scaler,

# Features (X), Targets (y) and indices (dates) over the whole time span for multi and single output respectively
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
X_train_multi_reshaped, X_val_multi_reshaped, X_test_multi_reshaped) = datasets[current_dataset]

hg_oi_statistics = calculate_moving_statistics(hg_oi_10y_df.index.tolist(), hg_oi_10y_df["T"], 4).rename(columns={"Value": "Order Intake"})
highlight_columns = ['Order Intake']
oi_plot, oi_plot_imgstream = create_plot(
    df=hg_oi_statistics[["Order Intake", "Moving Average", "Moving Std Dev"]].reset_index(drop=False), 
    highlight_columns=highlight_columns, 
    color_palette=color_palette,
    chart_title='Order Intake',
    xlabel='Datum',
    ylabel='Wert (in M€)',
    scale_factor=1e-6
)

results = compute_relationship_analysis(target_df=hg_oi_df[["T"]], predictors_df=oecd_df)
n_best = 10
best_predictors = results.reindex(results['Best Granger Causality P-Value'].sort_values(ascending=True).index).head(n_best)
best_correlations = results.reindex(results['Best Correlation'].abs().sort_values(ascending=False).index).head(n_best)
best_covariants = results.reindex(results['Best Covariance'].abs().sort_values(ascending=False).index).head(n_best)

predictors_plot, predictors_imgstream = plot_series_with_predictors(target_df=hg_oi_10y_df, predictors_df=oecd_df, best_predictors_df=best_predictors, n_best=10, title="Best Predictors for Order Intake", show=False)
best_predictors_plot_df = best_predictors[["Target", "Predictor", "Best Granger Causality P-Value", "Best Granger Causality P-Value Lag", "Granger Causality P-Values"]]
correlations_plot, correlations_imgstream = plot_series_with_predictors(target_df=hg_oi_10y_df, predictors_df=oecd_df, best_predictors_df=best_correlations, n_best=10, title="Strongest Correlating Indicators for Order Intake", show=False)
best_correlations[["Target", "Predictor", "Best Correlation", "Best Correlation Lag", "Correlations"]]
covariants_plot, covariance_imgstream = plot_series_with_predictors(target_df=hg_oi_10y_df, predictors_df=oecd_df, best_predictors_df=best_covariants, n_best=10, title="Strongest Covariant Indicators for Order Intake", show=False)
best_covariants[["Target", "Predictor", "Best Covariance", "Best Covariance Lag", "Covariances"]]

evaluation_nn, plot_nn = evaluate_model_multi_target(keras_load_model(os.path.join("Model Archive", f"OECD_NN.keras")), X_multi, y_multi, dates_multi, target_scaler, model_name='NN', plotly=True,show=False, return_plot=True)
evaluation_fcnn, plot_fcnn = evaluate_model_multi_target(keras_load_model(os.path.join("Model Archive", f"OECD_FFNN.keras")), X_multi, y_multi, dates_multi, target_scaler, model_name='FCNN', plotly=True,show=False, return_plot=True)


predictions_dict = {}
for dataset, data in datasets.items():
    (hg_oi_df, oecd_df, # Imported raw data
    time_series, target_single, features, targets_multi, # name of target and feature attributes/columns

    # Scaler
    feature_scaler,
    target_scaler,

    # Features (X), Targets (y) and indices (dates) over the whole time span for multi and single output respectively
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
    X_train_multi_reshaped, X_val_multi_reshaped, X_test_multi_reshaped) = data
    
    model_paths = {
        "FFNN": os.path.join("Model Archive", f"{dataset}_FFNN.keras"),
        "NN": os.path.join("Model Archive", f"{dataset}_NN.keras")
    }
    prediction_models = {}
    for model in model_paths.keys():
        prediction_models[model] = keras_load_model(model_paths[model])

    predictions = future_predictions(prediction_models, target_scaler, X_multi_all[len(X_multi_all)-1], dates_multi_all[len(dates_multi_all)-1])
    predictions_dict[dataset] = predictions
save_results_to_excel(predictions_dict.values(), "sales_predictions_06-24_from_07-24_to_06-25.xlsx", predictions_dict.keys(), [["Prediction Baseline", "Date", "Sales Prediction (NN)"]], [{"Sales Prediction (NN)": "Sales Prediction (Best Model)"}])

oi_df = hg_oi_df[["T"]]
oi_df.rename(columns={"T": "Order Intake"}, inplace=True)
predictions_df = pd.DataFrame(predictions_dict["OECD"][predictions_dict["OECD"].columns.drop("Prediction Baseline")])
predictions_df.set_index("Date", inplace=True)
predictions_df = pd.concat([predictions_df, oi_df], axis=0)
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


app.layout = [
    #dcc.Graph(figure=oi_plot),
    dcc.Graph(figure=predictors_plot),
    dcc.Graph(figure=correlations_plot),
    dcc.Graph(figure=covariants_plot),
    dcc.Graph(figure=plot_nn),
    dcc.Graph(figure=plot_fcnn),
    dcc.Graph(figure=oi_pred_plot)
]

if __name__ == '__main__':
    app.run(debug=True)