#%% Packages and Functions
import CSV
using DataFrames, Plots, MLJ; using Impute: srs; using StatsPlots: @df; using Chain: @chain

code_folder = "source"

include(joinpath(code_folder, "functions.jl"))

#%% Reading and Cleaning Data

dataset_dir = "datasets"

train = joinpath(dataset_dir, "train.csv") |> CSV.File |> DataFrame |> clean_data
test = joinpath(dataset_dir, "test.csv") |> CSV.File |> DataFrame |> clean_data

train |> schema

#%% Defining Model Pipelines

reg_model_names = [
    "ARD",
    "AdaBoost",
    "Bagging",
    "Decision Tree",
    "Elastic Net",
    "Epsilon SVR",
    "ExtraTrees",
    "Gradient Boosting",
    "KNN",
    "Random Forest",
    "Ridge",
    "SVM",
]

reg_model_pipelines = [
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load ARDRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load AdaBoostRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load BaggingRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load DecisionTreeRegressor pkg = DecisionTree verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load ElasticNetRegressor pkg = MLJLinearModels verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load EpsilonSVR pkg = LIBSVM verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load ExtraTreesRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load GradientBoostingRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load KNeighborsRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load RandomForestRegressor pkg = DecisionTree verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load RidgeRegressor pkg = MLJLinearModels verbosity = 0)()
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load SVMRegressor pkg = ScikitLearn verbosity = 0)()
    )
]

#%% Getting Training Data Ready

X_train = @chain train select(Not(:SalePrice))

y_train = float.(train[!, :SalePrice])

#%% Getting Baseline Results

baseline_results = @chain DataFrame(
    model = reg_model_names,
    cv_rmse = [
        evaluate(
            reg,
            X_train,
            y_train,
            resampling = CV(shuffle = true, nfolds = 5),
            verbosity = 0,
        ).measurement[1] for reg ∈ reg_model_pipelines
    ],
) begin
    sort(:cv_rmse)
end

@df baseline_results bar(
    :model,
    :cv_rmse,
    orientation = :h,
    yflip = true,
    legend = false,
    xlabel = "Cross-Validated RMSE",
    title = "Comparing Baseline Models",
)

# models I will tune: 
# 1. GradientBoostingRegressor
# 2. ExtraTrees
# 3. BaggingRegressor
# 4. RandomForestRegressor
# 5. ARD
# 6. Elastic Net