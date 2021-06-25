#%% Packages
using CSV, DataFrames, Statistics, Plots
using Impute: srs
using MLJ
using StatsPlots: @df
using Chain: @chain
using Pipe: @pipe

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

order_qual(cat_vector) = levels!(cat_vector, ["Po", "Fa", "TA", "Gd", "Ex"])

function reorder_levels(dataset)
    qual_cond_vars = [:ExterQual, :ExterCond, :BsmtQual, :BsmtCond, :HeatingQC, :KitchenQual, :GarageQual, :GarageCond]

    @chain dataset begin
        select(
            Symbol.(names(dataset)),
            :LotShape => (x -> levels!(x, ["Reg", "IR1", "IR2", "IR3"])) => :LotShape,
            :LandSlope => (x -> levels!(x, ["Gtl", "Mod", "Sev"])) => :LandSlope,
            :BsmtExposure => (x -> levels!(x, ["No", "Mn", "Av", "Gd"])) => :BsmtExposure,
            :Functional => (x -> levels!(x, ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"])) => :Functionality,
            qual_cond_vars .=> order_qual .=> qual_cond_vars
        )
    end
end

clean_data = reorder_levels ∘ finalize_types ∘ fix_lot_frontage ∘ dichotomize_some_features ∘ fix_dtypes ∘ drop_unnecessary_features

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

X = @chain train begin
    select(Not(:SalePrice))
    preprocess
end
y = float.(train[!, :SalePrice])

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
reg_pipelines = [
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load ARDRegressor pkg = ScikitLearn verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load AdaBoostRegressor pkg = ScikitLearn verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load BaggingRegressor pkg = ScikitLearn verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load DecisionTreeRegressor pkg = DecisionTree verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load ElasticNetRegressor pkg = MLJLinearModels verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load EpsilonSVR pkg = LIBSVM verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load ExtraTreesRegressor pkg = ScikitLearn verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load GradientBoostingRegressor pkg = ScikitLearn verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load KNeighborsRegressor pkg = ScikitLearn verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load RandomForestRegressor pkg = DecisionTree verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load RidgeRegressor pkg = MLJLinearModels verbosity = 0)()
        ),
    @pipeline(
            X -> srs(X),
            X -> coerce(X, :SalePrice => Continuous),
            OneHotEncoder(ordered_factor = false, drop_last = true),
            Standardizer(count = true),
            (@load SVMRegressor pkg = ScikitLearn verbosity = 0)()
        )
]

evaluate(reg_pipelines[2], X, y, resampling=CV(shuffle=true, nfolds=5), verbosity=0)

baseline_results = @chain DataFrame(
	model_name = reg_model_names,
	cv_acc = [MLJ.evaluate(reg, MLJ.table(Matrix(X)), y, resampling=CV(shuffle=true, nfolds=5),
					   verbosity=0).measurement[1]
			  for reg ∈ reg_pipelines]
) begin
    sort(_, :cv_acc, rev = true) 
end

eltype.(eachcol(train)) |> unique

num_vars = @chain train begin
    srs
    finalize_types
    dtype_info
    filter(x -> x.dtype ∈ [Int64, Float64], _)
    _[!, :col]
    Symbol.(_)
end

train_ready = @chain train begin
    srs
    finalize_types
end

# figure out which numeric varialbes have tiny variance, maybe drop them