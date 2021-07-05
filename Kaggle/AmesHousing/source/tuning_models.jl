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

train = @chain dataset_dir joinpath("train.csv") CSV.File DataFrame clean_data
test = @chain dataset_dir joinpath("test.csv") CSV.File DataFrame clean_data

X_train = select(train, Not([:SalePrice, :Id]))

y_train = float.(train[!, :SalePrice])

schema(train)

### Tuning Models

# models I will tune: 
# 1. GradientBoostingRegressor
# 2. ExtraTrees
# 3. BaggingRegressor
# 4. RandomForestRegressor
# 5. ARD
# 6. Elastic Net

#### GradientBoostingRegressor

#%% Tuning individual tree parameters

gbr_pipeline = @pipeline(
    # get data ready for modeling
    X -> ready_data(X, train),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load GradientBoostingRegressor pkg = ScikitLearn verbosity = 0)()
)

min_samples_split_range = range(
    gbr_pipeline,
    :(gradient_boosting_regressor.min_samples_split),
    lower = 2,
    upper = 10,
)

max_depth_range =
    range(gbr_pipeline, :(gradient_boosting_regressor.max_depth), lower = 3, upper = 10)

tuned_gbr_pipeline1 = @chain gbr_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 11),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [min_samples_split_range, max_depth_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_gbr_pipeline1)

gbr_params1 = fitted_params(tuned_gbr_pipeline1)
gbr_report1 = report(tuned_gbr_pipeline1)

gbr_params1.best_model
gbr_report1.best_history_entry # RMSE = $25,186.25

gbr_pipeline.gradient_boosting_regressor.min_samples_split =
    gbr_params1.best_model.gradient_boosting_regressor.min_samples_split # 4
gbr_pipeline.gradient_boosting_regressor.max_depth =
    gbr_params1.best_model.gradient_boosting_regressor.max_depth # 4

#%% Tuning Boosting parameters

loss_range = range(
    gbr_pipeline,
    :(gradient_boosting_regressor.loss),
    values = ["ls", "lad", "huber"],
)

learning_rate_range = range(
    gbr_pipeline,
    :(gradient_boosting_regressor.learning_rate),
    lower = 0.01,
    upper = 0.2
)

n_estimators_range = range(
    gbr_pipeline,
    :(gradient_boosting_regressor.n_estimators),
    lower = 50,
    upper = 150
)

tuned_gbr_pipeline2 = @chain gbr_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 5),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [loss_range, learning_rate_range, n_estimators_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_gbr_pipeline2)

gbr_params2 = fitted_params(tuned_gbr_pipeline2)
gbr_report2 = report(tuned_gbr_pipeline2)

gbr_params2.best_model
gbr_report2.best_history_entry # RMSE = $26,036.80

gbr_pipeline.gradient_boosting_regressor.loss =
    gbr_params2.best_model.gradient_boosting_regressor.loss # "ls"
gbr_pipeline.gradient_boosting_regressor.learning_rate =
    gbr_params2.best_model.gradient_boosting_regressor.learning_rate # 0.2
gbr_pipeline.gradient_boosting_regressor.n_estimators =
    gbr_params2.best_model.gradient_boosting_regressor.n_estimators # 125

#### ExtraTrees

#%% Tuning Individual Tree parameters

xt_pipeline = @pipeline(
    # get data ready for modeling
    X -> ready_data(X, train),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load ExtraTreesRegressor pkg = ScikitLearn verbosity = 0)()
)

min_samples_split_range =
    range(xt_pipeline, :(extra_trees_regressor.min_samples_split), lower = 2, upper = 10)
max_depth_range = range(Int, :(extra_trees_regressor.max_depth), lower = 2, upper = 15)

tuned_xt_pipeline1 = @chain xt_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 11),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [max_depth_range, min_samples_split_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_xt_pipeline1)

xt_params1 = fitted_params(tuned_xt_pipeline1)
xt_report1 = report(tuned_xt_pipeline1)

xt_params1.best_model
xt_report1.best_history_entry # RMSE = $29,630.59

xt_pipeline.extra_trees_regressor.min_samples_split =
    xt_params1.best_model.extra_trees_regressor.min_samples_split # 3
xt_pipeline.extra_trees_regressor.max_depth =
    xt_params1.best_model.extra_trees_regressor.max_depth # 10

#%% Tuning Ensemble Parameters

n_estimators_range =
    range(xt_pipeline, :(extra_trees_regressor.n_estimators), lower = 50, upper = 150)
max_features_range = range(
    xt_pipeline,
    :(extra_trees_regressor.max_features),
    values = ["sqrt", "log2", "auto"],
)

tuned_xt_pipeline2 = @chain xt_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 11),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [n_estimators_range, max_features_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_xt_pipeline2)

xt_params2 = fitted_params(tuned_xt_pipeline2)
xt_report2 = report(tuned_xt_pipeline2)

xt_params2.best_model
xt_report2.best_history_entry # RMSE = $29,779.51

xt_pipeline.extra_trees_regressor.n_estimators =
    xt_params2.best_model.extra_trees_regressor.n_estimators # 50
xt_pipeline.extra_trees_regressor.max_features =
    xt_params2.best_model.extra_trees_regressor.max_features # "auto"

#%% Bagging Regressor

br_pipeline = @pipeline(
    # get data ready for modeling
    X -> ready_data(X, train),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load BaggingRegressor pkg = ScikitLearn verbosity = 0)()
)

n_estimators_range =
    range(br_pipeline, :(bagging_regressor.n_estimators), lower = 5, upper = 20)
max_samples_range =
    range(br_pipeline, :(bagging_regressor.max_samples), lower = 0.5, upper = 1.0)
max_features_range =
    range(br_pipeline, :(bagging_regressor.max_features), lower = 0.3, upper = 1.0)

tuned_br_pipeline = @chain br_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 5),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [n_estimators_range, max_samples_range, max_features_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_br_pipeline)

br_params = fitted_params(tuned_br_pipeline)
br_report = report(tuned_br_pipeline)

br_params.best_model
br_report.best_history_entry # $29,040.57

br_pipeline.bagging_regressor.n_estimators =
    br_params.best_model.bagging_regressor.n_estimators # 20
br_pipeline.bagging_regressor.max_samples =
    br_params.best_model.bagging_regressor.max_samples # 1.0
br_pipeline.bagging_regressor.max_features =
    br_params.best_model.bagging_regressor.max_features # 1.0

#### Random Forest Regressor

#%% Tuning Individual Tree Parameters

rf_pipeline = @pipeline(
    # get data ready for modeling
    X -> ready_data(X, train),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load RandomForestRegressor pkg = DecisionTree verbosity = 0)()
)

max_depth_range = range(
    rf_pipeline,
    :(random_forest_regressor.max_depth),
    values = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)
min_samples_split_range =
    range(rf_pipeline, :(random_forest_regressor.min_samples_split), lower = 2, upper = 10)

tuned_rf_pipeline1 = @chain rf_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 11),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [max_depth_range, min_samples_split_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_rf_pipeline1)

rf_params1 = fitted_params(tuned_rf_pipeline1)
rf_report1 = report(tuned_rf_pipeline1)

rf_params1.best_model
rf_report1.best_history_entry # $32,475.51

rf_pipeline.random_forest_regressor.max_depth =
    rf_params1.best_model.random_forest_regressor.max_depth # -1
rf_pipeline.random_forest_regressor.min_samples_split =
    rf_params1.best_model.random_forest_regressor.min_samples_split # 2

#%% Tuning Ensemble Parameters

n_subfeatures_range = range(
    rf_pipeline,
    :(random_forest_regressor.n_subfeatures),
    values = [
        -1,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        170,
        190,
        200,
    ],
)
n_trees_range =
    range(rf_pipeline, :(random_forest_regressor.n_trees), lower = 5, upper = 100)
sampling_fraction_range = range(
    rf_pipeline,
    :(random_forest_regressor.sampling_fraction),
    lower = 0.4,
    upper = 1.0,
)

tuned_rf_pipeline2 = @chain rf_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 5),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [n_subfeatures_range, n_trees_range, sampling_fraction_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_rf_pipeline2)

rf_params2 = fitted_params(tuned_rf_pipeline2)
rf_report2 = report(tuned_rf_pipeline2)

rf_params2.best_model
rf_report2.best_history_entry # $28,489.60

rf_pipeline.random_forest_regressor.n_subfeatures = 
    rf_params2.best_model.random_forest_regressor.n_subfeatures # 130
rf_pipeline.random_forest_regressor.n_trees = 
    rf_params2.best_model.random_forest_regressor.n_trees # 52
rf_pipeline.random_forest_regressor.sampling_fraction = 
    rf_params2.best_model.random_forest_regressor.sampling_fraction # 1.0

#### ARDRegressor

#%% Tuning number of iterations

ard_pipeline = @pipeline(
    # get data ready for modeling
    X -> ready_data(X, train),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load ARDRegressor pkg = ScikitLearn verbosity = 0)()
)

n_iter_range = range(ard_pipeline, :(ard_regressor.n_iter), lower = 100, upper = 500)


tuned_ard_pipeline1 = @chain ard_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 50),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [n_iter_range]
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_ard_pipeline1)

ard_params1 = fitted_params(tuned_ard_pipeline1)
ard_report1 = report(tuned_ard_pipeline1)

ard_params1.best_model
ard_report1.best_history_entry # $31,379.92

ard_pipeline.ard_regressor.n_iter =
    ard_params1.best_model.ard_regressor.n_iter # 

#%% Tuning beta priors

α1_range = range(ard_pipeline, :(ard_regressor.alpha_1), lower = 0.000001, upper = 0.0001)
α2_range = range(ard_pipeline, :(ard_regressor.alpha_2), lower = 0.000001, upper = 0.0001)
λ1_range = range(ard_pipeline, :(ard_regressor.lambda_1), lower = 0.000001, upper = 0.0001)
λ2_range = range(ard_pipeline, :(ard_regressor.lambda_2), lower = 0.000001, upper = 0.0001)

tuned_ard_pipeline2 = @chain ard_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 4),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [α1_range, α2_range, λ1_range, λ2_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_ard_pipeline2)

ard_params2 = fitted_params(tuned_ard_pipeline2)
ard_report2 = report(tuned_ard_pipeline2)

ard_params2.best_model
ard_report2.best_history_entry # $32,089.25

ard_pipeline.ard_regressor.alpha_1 =
    ard_params2.best_model.ard_regressor.alpha_1 # 
ard_pipeline.ard_regressor.alpha_2 =
    ard_params2.best_model.ard_regressor.alpha_2 # 
ard_pipeline.ard_regressor.lambda_1 =
    ard_params2.best_model.ard_regressor.lambda_1 # 
ard_pipeline.ard_regressor.lambda_2 =
    ard_params2.best_model.ard_regressor.lambda_2 # 


#%% Tuning Elastic Net

en_pipeline = @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load ElasticNetRegressor pkg = MLJLinearModels verbosity = 0)()
    )

tuned_en_pipeline = @chain en_pipeline begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 5),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [_range, _range, _range]
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_en_pipeline)

en_params = fitted_params(tuned_en_pipeline)
en_report = report(tuned_en_pipeline)

en_params.best_model # 
en_report.best_history_entry # $


#%% Manually Defining Best Models

### If you haven't already tuned all models in this session

model_names = [
    "GradientBoostingRegressor",
    "ExtraTrees",
    "BaggingRegressor",
    "RandomForestRegressor",
    "ARDRegressor"
]

tuned_pipelines = [
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
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
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load ExtraTreesRegressor pkg = ScikitLearn verbosity = 0)(
            min_samples_split = 3,
            max_depth = 10,
            n_estimators = 50,
            max_features = "auto"
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load BaggingRegressor pkg = ScikitLearn verbosity = 0)(
            n_estimators = 20,
            max_samples = 1.0,
            max_features = 1.0
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
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
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, train),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load ARDRegressor pkg = ScikitLearn verbosity = 0)(
            n_iter = 124
        )
    )
]

cv_rmses = [
    evaluate(
        reg,
        X_train,
        y_train,
        resampling = CV(shuffle = true, nfolds = 5),
        verbosity = 0,
    ).measurement[1] for reg ∈ tuned_pipelines
]

#%% Defining Results

#### If you have already tuned all models in this session

cv_rmses = [
    gbr_report2.best_history_entry.measurement[1],
    xt_report2.best_history_entry.measurement[1],
    br_report.best_history_entry.measurement[1],
    rf_report2.best_history_entry.measurement[1],
    ard_report2.best_history_entry.measurement[1]
]

#%% Gathering Results

tuned_results = @chain DataFrame(
    model = model_names,
    cv_rmse = cv_rmses
) sort(:cv_rmse)

@df tuned_results bar(
    :model,
    :cv_rmse,
    orientation = :h,
    yflip = true,
    legend = false,
    xlabel = "Cross-Validated RMSE",
    title = "Comparing Tuned Models",
)

#%% Saving Best Models

model_dir = "models"

@chain tuned_pipelines[1] begin
    machine(X_train, y_train)
    fit!
    MLJ.save(joinpath(model_dir, "gbr.jlso"), _)
end

@chain tuned_pipelines[4] begin
    machine(X_train, y_train)
    fit!
    MLJ.save(joinpath(model_dir, "rf.jlso"), _)
end