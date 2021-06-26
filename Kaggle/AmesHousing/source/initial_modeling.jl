#%% Packages
using CSV, DataFrames, Statistics, Plots
using Impute: srs
using MLJ
using StatsPlots: @df
using Chain: @chain

#%% Reading and Cleaning Data

dtype_info(dataframe) = DataFrame(
    col = names(dataframe),
    dtype = eltype.(eachcol(dataframe))
)

has_value(value) = ismissing(value) ? 0.0 : 1.0

parse_int(value) = ismissing(value) ? missing : parse(Int64, value)

# convert "NA" string values to missings
function fix_na(input_vec)
    input_type = eltype(input_vec)
    vector = convert(Vector{Union{Missing, input_type}}, input_vec)
    vector[vector .== "NA"] .= missing
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
        select(Symbol.(names(dataset)),
            # convert string cols that should be numeric
            should_be_numeric .=> (x -> parse_int.(x)) .=> should_be_numeric,
            # convert int cols that should be categorical
            should_be_factor .=> (x -> string.(x)) .=> should_be_factor
        )
    end

    # get names of all categorical columns
    cat_vars = @chain out_df begin
        dtype_info
        filter(x -> x.dtype ∈ [String, Union{Missing, String}], _)
        _[!, :col]
        Symbol.(_)
    end

    # convert all String columns to categorical
    @chain out_df begin
        select(Symbol.(names(out_df)),
            cat_vars .=> categorical .=> cat_vars
            )
    end
end

# dichotomizes the 5 features with the most missing values
function dichotomize_some_features(dataset)
    @chain dataset begin
        select(Not([:PoolQC, :MiscFeature, :Alley, :Fence]),
            :PoolQC => (x -> has_value.(x)) => :HasPool,
            :MiscFeature => (x -> has_value.(x)) => :HasMiscFeature,
            :Alley => (x -> has_value.(x)) => :HasAlley,
            :Fence => (x -> has_value.(x)) => :HasFence
        )
    end
end

function fix_lot_frontage(dataset)
    @chain dataset begin
        select(Symbol.(names(dataset)),
        # replace missings with 0.0
        :LotFrontage => (x -> coalesce.(x, 0)) => :LotFrontage
        )
    end
end

function remove_missing_union(col_vector)
    if (eltype(col_vector) == Union{Missing, CategoricalValue{String, UInt32}}) && sum(ismissing.(col_vector)) == 0
        convert(CategoricalArray, string.(col_vector))
    elseif eltype(col_vector) == Union{Missing, Int64} && sum(ismissing.(col_vector)) == 0
        convert.(Int64, col_vector)
    else
        col_vector
    end
end

function finalize_types(dataset)
    all_cols = Symbol.(names(dataset))

    @chain dataset begin
        select(
            all_cols .=> remove_missing_union .=> all_cols
        )
    end
end


#clean_data = set_factor_levels ∘ finalize_types ∘ fix_lot_frontage ∘ dichotomize_some_features ∘ fix_dtypes ∘ drop_unnecessary_features
clean_data = finalize_types ∘ fix_lot_frontage ∘ dichotomize_some_features ∘ fix_dtypes ∘ drop_unnecessary_features

dataset_dir = "datasets"

train = joinpath(dataset_dir, "train.csv") |> CSV.File |> DataFrame |> clean_data
test = joinpath(dataset_dir, "test.csv") |> CSV.File |> DataFrame |> clean_data

train |> schema

#%% Data Preprocessing

function preprocess(dataset)
    #ordered_cat_vars = [:LotShape, :LandSlope, :BsmtExposure, :Functional,
                        #:ExterQual, :ExterCond, :BsmtQual, :BsmtCond,
                        #:HeatingQC, :KitchenQual, :GarageQual, :GarageCond]

    dataset_ready = @chain dataset srs finalize_types

    cat_encoder = @chain OneHotEncoder(
        ordered_factor = false, drop_last = true
        ) begin
            machine(dataset_ready)
            MLJ.fit!
    end

    standardizer = @chain Standardizer(
        count = true
        ) begin
        machine(dataset_ready)
        MLJ.fit!
    end

    @chain dataset_ready begin
        MLJ.transform(standardizer, _)
        MLJ.transform(cat_encoder, _)
    end
end

#%% Baseline Modeling Results

# reg_pipelines = [
#     (@load ARDRegressor pkg = ScikitLearn verbosity = 0)(),
#     (@load AdaBoostRegressor pkg = ScikitLearn verbosity = 0)(),
#     (@load BaggingRegressor pkg = ScikitLearn verbosity = 0)(),
#     (@load DecisionTreeRegressor pkg = DecisionTree verbosity = 0)(),
#     (@load ElasticNetRegressor pkg = MLJLinearModels verbosity = 0)(),
#     (@load EpsilonSVR pkg = LIBSVM verbosity = 0)(),
#     (@load ExtraTreesRegressor pkg = ScikitLearn verbosity = 0)(),
#     (@load GradientBoostingRegressor pkg = ScikitLearn verbosity = 0)(),
#     (@load KNeighborsRegressor pkg = ScikitLearn verbosity = 0)(),
#     (@load RandomForestRegressor pkg = DecisionTree verbosity = 0)(),
#     (@load RidgeRegressor pkg = MLJLinearModels verbosity = 0)(),
#     (@load SVMRegressor pkg = ScikitLearn verbosity = 0)()
# ]

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
    "SVM"
]

ready_data(dataframe) = @chain dataframe srs finalize_types coerce(:SalePrice => Continuous) match_factor_levels(train)

reg_model_pipelines = [
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load ARDRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load AdaBoostRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load BaggingRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load DecisionTreeRegressor pkg = DecisionTree verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load ElasticNetRegressor pkg = MLJLinearModels verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load EpsilonSVR pkg = LIBSVM verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load ExtraTreesRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load GradientBoostingRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load KNeighborsRegressor pkg = ScikitLearn verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load RandomForestRegressor pkg = DecisionTree verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load RidgeRegressor pkg = MLJLinearModels verbosity = 0)()
    ),
    @pipeline(
            # get data ready for modeling
            X -> ready_data(X),
            # Standardize numeric variables
            Standardizer(count = true),
            # Dummify categorical variables
            OneHotEncoder(ordered_factor = false, drop_last = true),
            # load into model
            (@load SVMRegressor pkg = ScikitLearn verbosity = 0)()
    )
]

function match_factor_levels(dataset, full_data)
    # Don't fix PoolQC, MiscFeature, Alley, or Fence

    # first, get names of categorical variables (minus BsmtCond and MSSubClass since those are special cases)
    cat_vars = @chain dataset begin
        dtype_info
        filter(x -> x.dtype ∈ [CategoricalValue{String, UInt32}, Union{Missing, CategoricalValue{String, UInt32}}] && x.col ∉ ["BsmtCond", "MSSubClass"], _)
        _[!, :col]
        Symbol.(_)
    end

    # fix BsmtCond and MSSubClass levels
    out_df = @chain dataset begin
        select(
            Symbol.(names(dataset)),
            :BsmtCond => (x -> levels!(x, ["Po", "Fa", "TA", "Gd", "Ex"])) => :BsmtCond,
            :MSSubClass => (x -> levels!(x, ["20", "30", "40", "45", "50", "60", "70", "75", "80", "85", "90", "120", "150", "160", "180", "190"])) => :MSSubClass
        )
    end

    # for every categorical variable
    for varname ∈ cat_vars
        levels!(out_df[!, varname], levels(full_data[!, varname]))
    end

    return out_df
end

X_train = @chain train select(Not(:SalePrice))

y_train = float.(train[!, :SalePrice])

baseline_results = DataFrame(
    model = reg_model_names,
    cv_rmse = [
        evaluate(reg, X_train, y_train, resampling=CV(shuffle=true, nfolds=5), verbosity=0).measurement[1]
        for reg ∈ reg_model_pipelines
    ]
)

baseline_results = @chain baseline_results sort(_, :cv_rmse)

@df baseline_results bar(
	:model,
	:cv_rmse,
	orientation = :h,
	yflip = true,
	legend = false,
	xlabel = "Cross-Validated RMSE",
	title = "Comparing Baseline Models"
)

# models I will tune: 
    # 1. GradientBoostingRegressor
    # 2. ExtraTrees
    # 3. BaggingRegressor
    # 4. RandomForestRegressor
    # 5. ARD