eltype_info(dataframe) = DataFrame(
    col = names(dataframe),
    element_type = eltype.(eachcol(dataframe))
)

type_info(dataframe) = DataFrame(
    col = names(dataframe),
    overall_type = typeof.(eachcol(dataframe))
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

drop_unnecessary_features(dataset) = @chain dataset select(Not([:FireplaceQu]))

# first fix NAs, then convert some to numeric, make factors categorical
function fix_dtypes(dataset)
    if "SalePrice" ∈ names(dataset)
        should_be_numeric = [:MasVnrArea, :LotFrontage, :GarageYrBlt]
    else
        should_be_numeric = [
            :MasVnrArea, :LotFrontage, :GarageYrBlt,
            :BsmtFinSF1, :TotalBsmtSF, :GarageArea,
            :BsmtFinSF2, :BsmtUnfSF, :BsmtFullBath,
            :BsmtHalfBath, :GarageCars]
    end
    
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
        eltype_info
        filter(x -> x.element_type ∈ [String, Union{Missing, String}], _)
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
    if typeof(col_vector) == CategoricalArrays.CategoricalVector{Union{Missing, String}, UInt32, String, CategoricalArrays.CategoricalValue{String, UInt32}, Missing}
        CategoricalArray(string.(col_vector))
    elseif typeof(col_vector) == Vector{Union{Missing, Int64}}
        convert(Vector{Int64}, col_vector)
    else
        col_vector
    end
end

function finalize_types(dataset)
    cols_marked_missing = @chain dataset begin
        type_info
        filter(x -> x.overall_type ∉ [Vector{Float64}, Vector{Int64}, CategoricalArrays.CategoricalVector{String, UInt32, String, CategoricalArrays.CategoricalValue{String, UInt32}, Union{}}], _)
        _[!, :col]
        Symbol.(_)
    end

    cols_actually_missing = @chain dataset begin
        missing_per_column
        _[!, :column]
        Symbol.(_)
    end

    cols_need_fixing = setdiff(cols_marked_missing, cols_actually_missing)

    @chain dataset begin
        select(Symbol.(names(dataset)),
            cols_need_fixing .=> disallowmissing .=> cols_need_fixing
        )
    end
end

function levels_per_factor(dataframe)
    cat_vars = @chain dataframe begin
        eltype_info
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

clean_data = finalize_types ∘ fix_lot_frontage ∘ dichotomize_some_features ∘ fix_dtypes ∘ drop_unnecessary_features

function match_factor_levels(dataset, full_data)
    # Don't fix PoolQC, MiscFeature, Alley, or Fence

    # first, get names of categorical variables (minus BsmtCond and MSSubClass since those are special cases)
    cat_vars = @chain dataset begin
        type_info
        filter(x -> x.overall_type ∉ [Vector{Int64}, Vector{Float64}, Vector{Union{Missing, Int64}}] && x.col ∉ ["BsmtCond", "MSSubClass"], _)
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

ready_data(dataframe, full_dataframe) =
    @chain dataframe srs disallowmissing coerce(:SalePrice => Continuous) match_factor_levels(
        full_dataframe
    )


function generate_submission(model_pipeline_fit, new_data)
    # remove Id from predictors
    X = new_data[!, Not(:Id)]

    if "SalePrice" ∈ names(X) 
        X = X[!, Not(:SalePrice)]
    end

    DataFrame(
        Id = new_data[!, :Id],
        SalePrice = predict(model_pipeline_fit, X)
    )
end

mutable struct MultiLayerBuilder <: MLJFlux.Builder
    n_hidden_layers :: Int
    hidden_layer_width :: Int
end

function MLJFlux.build(nn::MultiLayerBuilder, rng, n_in, n_out)
	init = Flux.glorot_uniform(rng)

    # initialize list of layers with input layer
    layer_list = [Dense(n_in, nn.hidden_layer_width, init = init)]
    
    # add hidden layers
    for _ ∈ 1:nn.n_hidden_layers
        push!(layer_list, Dense(nn.hidden_layer_width, nn.hidden_layer_width, init = init))
    end

    # place output layer at the end
    push!(layer_list, Dense(nn.hidden_layer_width, n_out, init = init))

	return Chain(layer_list...)
end

println("Functions imported")