#%% Packages and Functions
using CSV, DataFrames, Statistics, Plots
using MLJ: schema
using StatsPlots: @df
using Chain: @chain
using Pipe: @pipe

code_folder = "source"

#%% Reading and Cleaning Data

dataset_dir = "datasets"

train = joinpath(dataset_dir, "train.csv") |> CSV.File |> DataFrame |> clean_data
test = joinpath(dataset_dir, "test.csv") |> CSV.File |> DataFrame |> clean_data

train |> schema

#%% Seeing how many levels there are of each factor

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
    xlabel = "SalePrice (\$)",
    ylabel = "Count",
    title = "Distribution of SalePrice in Training Data",
    legend = false
)