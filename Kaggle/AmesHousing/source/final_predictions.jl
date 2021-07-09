#%% Packages
import CSV
using DataFrames, Plots, MLJ, CategoricalArrays
using Impute: srs
using StatsPlots: @df
using Chain: @chain

code_folder = "source"

include(joinpath(code_folder, "functions.jl"))

#%% Reading and Cleaning Data

dataset_dir = "datasets"

df = @chain dataset_dir joinpath("train.csv") CSV.File DataFrame clean_data
test = @chain dataset_dir joinpath("test.csv") CSV.File DataFrame clean_data

X_data = select(df, Not([:SalePrice, :Id]))

y_data = float.(df[!, :SalePrice])

#%% Reading in and Fitting Best Models

model_dir = "models"

nn_fit = @chain @pipeline(
    # get data ready for modeling
    X -> ready_data(X, X_data),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
        builder = MultiLayerBuilder(4, 25)
    )
) begin
    machine(X_data, y_data)
    fit!
end

gbr_fit = @chain @pipeline(
    # get data ready for modeling
    X -> ready_data(X, X_data),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load GradientBoostingRegressor pkg = ScikitLearn verbosity = 0)(
        min_samples_split = 4,
        max_depth = 4,
        loss = "ls",
        learning_rate = 0.2,
        n_estimators = 125
    )
) begin
    machine(X_data, y_data)
    fit!
end

rf_fit = @chain @pipeline(
    # get data ready for modeling
    X -> ready_data(X, X_data),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load RandomForestRegressor pkg = DecisionTree verbosity = 0)(
        min_samples_split = 2,
        max_depth = -1,
        n_subfeatures = 130,
        n_trees = 52,
        sampling_fraction = 1.0
    )
) begin
    machine(X_data, y_data)
    fit!
end

#%% Generating Submissions

submission_dir = "submissions"

@chain nn_fit begin
    generate_submission(test)
    CSV.write(joinpath(submission_dir, "submission_nn.csv"), _)
end

@chain gbr_fit begin
    generate_submission(test)
    CSV.write(joinpath(submission_dir, "submission_gbr.csv"), _)
end

@chain rf_fit begin
    generate_submission(test)
    CSV.write(joinpath(submission_dir, "submission_rf.csv"), _)
end