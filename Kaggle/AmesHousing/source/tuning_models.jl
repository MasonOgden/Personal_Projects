#%% Packages
using CSV, DataFrames, Statistics, Plots, MLJ
using Impute: srs
using StatsPlots: @df
using Chain: @chain

#%% Reading and Cleaning Data

dtype_info(dataframe) =
    DataFrame(col = names(dataframe), dtype = eltype.(eachcol(dataframe)))

has_value(value) = ismissing(value) ? 0.0 : 1.0

parse_int(value) = ismissing(value) ? missing : parse(Int64, value)

# convert "NA" string values to missings
function fix_na(input_vec)
    input_type = eltype(input_vec)
    vector = convert(Vector{Union{Missing,input_type}}, input_vec)
    vector[vector.=="NA"] .= missing
    return vector
end

function drop_unnecessary_features(dataset)
    @chain dataset select(Not([:Id, :FireplaceQu]))
end

# first fix NAs, then convert some to numeric, make factors categorical
function fix_dtypes(dataset)
    should_be_numeric = [:MasVnrArea, :LotFrontage, :GarageYrBlt]
    should_be_factor = [:MSSubClass, :MoSold, :YrSold]

    out_df = @chain dataset begin
        # replace "NA" String values with missing
        mapcols(fix_na, _)
        select(
            Symbol.(names(dataset)),
            # convert string cols that should be numeric
            should_be_numeric .=> (x -> parse_int.(x)) .=> should_be_numeric,
            # convert int cols that should be categorical
            should_be_factor .=> (x -> string.(x)) .=> should_be_factor,
        )
    end

    # get names of all categorical columns
    cat_vars = @chain out_df begin
        dtype_info
        filter(x -> x.dtype ∈ [String, Union{Missing,String}], _)
        _[!, :col]
        Symbol.(_)
    end

    # convert all String columns to categorical
    @chain out_df begin
        select(Symbol.(names(out_df)), cat_vars .=> categorical .=> cat_vars)
    end
end

# dichotomizes the 5 features with the most missing values
function dichotomize_some_features(dataset)
    @chain dataset begin
        select(
            Not([:PoolQC, :MiscFeature, :Alley, :Fence]),
            :PoolQC => (x -> has_value.(x)) => :HasPool,
            :MiscFeature => (x -> has_value.(x)) => :HasMiscFeature,
            :Alley => (x -> has_value.(x)) => :HasAlley,
            :Fence => (x -> has_value.(x)) => :HasFence,
        )
    end
end

function fix_lot_frontage(dataset)
    @chain dataset begin
        select(
            Symbol.(names(dataset)),
            # replace missings with 0.0
            :LotFrontage => (x -> coalesce.(x, 0)) => :LotFrontage,
        )
    end
end

function remove_missing_union(col_vector)
    if (eltype(col_vector) == Union{Missing,CategoricalValue{String,UInt32}}) &&
       sum(ismissing.(col_vector)) == 0
        convert(CategoricalArray, string.(col_vector))
    elseif eltype(col_vector) == Union{Missing,Int64} && sum(ismissing.(col_vector)) == 0
        convert.(Int64, col_vector)
    else
        col_vector
    end
end

function finalize_types(dataset)
    all_cols = Symbol.(names(dataset))

    @chain dataset begin
        select(all_cols .=> remove_missing_union .=> all_cols)
    end
end


#clean_data = set_factor_levels ∘ finalize_types ∘ fix_lot_frontage ∘ dichotomize_some_features ∘ fix_dtypes ∘ drop_unnecessary_features
clean_data =
    finalize_types ∘ fix_lot_frontage ∘ dichotomize_some_features ∘ fix_dtypes ∘
    drop_unnecessary_features

dataset_dir = "datasets"

train = joinpath(dataset_dir, "train.csv") |> CSV.File |> DataFrame |> clean_data
test = joinpath(dataset_dir, "test.csv") |> CSV.File |> DataFrame |> clean_data

X_train = select(train, Not(:SalePrice))

y_train = float.(train[!, :SalePrice])

train |> schema

### Tuning Models

ready_data(dataframe) =
    @chain dataframe srs finalize_types coerce(:SalePrice => Continuous) match_factor_levels(
        train,
    )

function match_factor_levels(dataset, full_data)
    # Don't fix PoolQC, MiscFeature, Alley, or Fence

    # first, get names of categorical variables (minus BsmtCond and MSSubClass since those are special cases)
    cat_vars = @chain dataset begin
        dtype_info
        filter(
            x ->
                x.dtype ∈ [
                    CategoricalValue{String,UInt32},
                    Union{Missing,CategoricalValue{String,UInt32}},
                ] && x.col ∉ ["BsmtCond", "MSSubClass"],
            _,
        )
        _[!, :col]
        Symbol.(_)
    end

    # fix BsmtCond and MSSubClass levels
    out_df = @chain dataset begin
        select(
            Symbol.(names(dataset)),
            :BsmtCond => (x -> levels!(x, ["Po", "Fa", "TA", "Gd", "Ex"])) => :BsmtCond,
            :MSSubClass =>
                (
                    x -> levels!(
                        x,
                        [
                            "20",
                            "30",
                            "40",
                            "45",
                            "50",
                            "60",
                            "70",
                            "75",
                            "80",
                            "85",
                            "90",
                            "120",
                            "150",
                            "160",
                            "180",
                            "190",
                        ],
                    )
                ) => :MSSubClass,
        )
    end

    # for every categorical variable
    for varname ∈ cat_vars
        levels!(out_df[!, varname], levels(full_data[!, varname]))
    end

    return out_df
end

#### GradientBoostingRegressor

#%% Tuning individual tree parameters

gbr_pipeline = @pipeline(
    # get data ready for modeling
    X -> ready_data(X),
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
gbr_report1.best_history_entry # RMSE = $25,641.02

gbr_pipeline.gradient_boosting_regressor.min_samples_split =
    gbr_params1.best_model.gradient_boosting_regressor.min_samples_split # 2
gbr_pipeline.gradient_boosting_regressor.max_depth =
    gbr_params1.best_model.gradient_boosting_regressor.max_depth # 3

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
    upper = 0.2,
)
n_estimators_range = range(
    gbr_pipeline,
    :(gradient_boosting_regressor.n_estimators),
    lower = 50,
    upper = 150,
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
    X -> ready_data(X),
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
    X -> ready_data(X),
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
    X -> ready_data(X),
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
rf_report1.best_history_entry # $33,697.57

rf_pipeline.random_forest_regressor.max_depth =
    rf_params.best_model.random_forest_regressor.max_depth # -1
rf_pipeline.random_forest_regressor.min_samples_split =
    rf_params.best_model.random_forest_regressor.min_samples_split # 2

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
rf_report2.best_history_entry # $33,697.57

#### ARDRegressor

#%% Tuning number of iterations

ard_pipeline = @pipeline(
    # get data ready for modeling
    X -> ready_data(X),
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
        tuning = Grid(resolution = 100),
        resampling = CV(shuffle = true, nfolds = 5),
        ranges = [n_iter_range],
    )
    machine(X_train, y_train)
end

MLJ.fit!(tuned_ard_pipeline1)

ard_params1 = fitted_params(tuned_ard_pipeline1)
ard_report1 = report(tuned_ard_pipeline1)

ard_params1.best_model
ard_report1.best_history_entry # $,.

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
ard_report2.best_history_entry # $,.

ard_pipeline.ard_regressor.alpha_1 =
    ard_params2.best_model.ard_regressor.alpha_1 # 
ard_pipeline.ard_regressor.alpha_2 =
    ard_params2.best_model.ard_regressor.alpha_2 # 
ard_pipeline.ard_regressor.lambda_1 =
    ard_params2.best_model.ard_regressor.lambda_1 # 
ard_pipeline.ard_regressor.lambda_2 =
    ard_params2.best_model.ard_regressor.lambda_2 # 


tuned_results = @chain DataFrame(
    model = [
        "GradientBoostingRegressor",
        "ExtraTrees",
        "BaggingRegressor",
        "RandomForestRegressor",
        "ARDRegressor"
    ],
    cv_rmse = [
        gbr_report2.best_history_entry.measurement[1],
        xt_report2.best_history_entry.measurement[1],
        br_report.best_history_entry.measurement[1],
        rf_report2.best_history_entry.measurement[1],
        ard_report2.best_history_entry.measurement[1]
    ]
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