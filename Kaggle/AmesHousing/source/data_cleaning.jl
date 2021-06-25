#%% Packages
using CSV, DataFrames, Statistics, Plots
using MLJ: schema
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
        convert(CategoricalArray, col_vector)
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

clean_data = finalize_types ∘ fix_lot_frontage ∘ dichotomize_some_features ∘ fix_dtypes ∘ drop_unnecessary_features

dataset_dir = "datasets"

train = joinpath(dataset_dir, "train.csv") |> CSV.File |> DataFrame |> clean_data
test = joinpath(dataset_dir, "test.csv") |> CSV.File |> DataFrame |> clean_data

train |> schema

#%% Seeing how many levels there are of each factor

function levels_per_factor(dataframe)
    cat_vars = @chain dataframe begin
        dtype_info
        filter(x -> x.dtype ∈ [Union{Missing, CategoricalValue{String, UInt32}}, CategoricalValue{String, UInt32}], _)
        _[!, :col]
    end
    
    @chain DataFrame(
        factor = cat_vars,
        num_levels = [size(levels(dataframe[:, col]), 1) for col ∈ cat_vars]
    ) begin 
    sort(:num_levels, rev = true)
    end
end

function missing_per_column(dataframe)
    num_missing(colname) = dataframe[:, colname] .|> ismissing |> sum

    @chain DataFrame(column = names(dataframe),
                    num_missing = dataframe |> names .|> num_missing
                    ) begin
                filter(row -> row.num_missing > 0, _)
                select([:column, :num_missing],
                    :num_missing => (x -> x ./ size(dataframe, 1)) => :prop_missing
                )
                sort(:prop_missing, rev = true)
                end
end

num_indicator_vars(dataframe) = @chain dataframe begin
    levels_per_factor
    _[!, :num_levels]
    (x -> x .- 1)(_)
    sum
end

@chain train begin
    levels_per_factor
    first(5)
end

n_indicator_vars = train |> num_indicator_vars

#%% Investigating Missing Values

missing_df = train |> missing_per_column

#%% Looking at Response variable

@df train histogram(
    :SalePrice,
    #bins = 30,
    xlabel = "SalePrice (\$)",
    ylabel = "Count",
    title = "Distribution of SalePrice in Training Data",
    legend = false
)