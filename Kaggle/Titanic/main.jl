#-----Packages-----#
using CSV, DataFrames, StatsModels, GLM, Statistics, StatsBase

#-----Data-----#
cd("C:/Users/mogde/Desktop/Git Repositories/Personal_Projects/Kaggle/Titanic")
train = DataFrame(CSV.File("datasets/train.csv"))
test = DataFrame(CSV.File("datasets/test.csv"))

#-----Data Cleaning-----#

num_missing(colname) = train[:, colname] .|> ismissing |> sum

missing_df = DataFrame(column = train |> names,
                       num_missing = train |> names .|> num_missing
                      )

function process_data(in_data, training_info)
    dataframe = select(
        in_data, Not([:PassengerId, :Name, :Ticket, :Cabin])
        )

    # convert strings to categories, SibSp and dParchto float
    transform!(dataframe, :Pclass => categorical, renamecols=false)
    transform!(dataframe, :Sex => categorical, renamecols=false)
    transform!(dataframe, :Embarked => categorical, renamecols=false)
    dataframe[!, :SibSp] = convert.(AbstractFloat, dataframe[:, :SibSp])
    dataframe[!, :Parch] = convert.(AbstractFloat, dataframe[:, :Parch])

    # impute columns with missing values
    dataframe[:, :Age] = coalesce.(dataframe[:, :Age], training_info["age_median"])
    dataframe[:, :Embarked] = coalesce.(dataframe[:, :Embarked], training_info["embarked_mode"])

    # scaling numeric
    dataframe[:, :Age] = (dataframe[:, :Age] .- training_info["age_params"][1]) ./ training_info["age_params"][2]
    dataframe[:, :SibSp] = (dataframe[:, :SibSp] .- training_info["sibsp_params"][1]) ./ training_info["sibsp_params"][2]
    dataframe[:, :Parch] = (dataframe[:, :Parch] .- training_info["parch_params"][1]) ./ training_info["parch_params"][2]
    dataframe[:, :Fare] = (dataframe[:, :Fare] .- training_info["fare_params"][1]) ./ training_info["fare_params"][2]
    return dataframe
end


training_info = Dict(
    "age_median" => train[:, :Age] |> skipmissing |> median,
    "embarked_mode" => train[:, :Embarked] |> skipmissing |> mode |> string,
    "age_params" => (coalesce.(train[:, :Age], train[:, :Age] |> skipmissing |> median) |> mean,
                     coalesce.(train[:, :Age], train[:, :Age] |> skipmissing |> median) |> std),
    "sibsp_params" => (train[:, :SibSp] |> mean, train[:, :SibSp] |> std),
    "parch_params" => (train[:, :Parch] |> mean, train[:, :Parch] |> std),
    "fare_params" => (train[:, :Fare] |> mean, train[:, :Fare] |> std)
)

#-----Making Cross-Validation Folds-----#
function get_cross_val_splits(data, k)
    # all indices
    all_indices = Set{Int}(1:nrow(data))
    # indices that aren't in a fold yet
    indices_remaining = Set(1:nrow(data))
    # number of indices per fold
    fold_size = convert(Int, floor(nrow(data) / k))
    # will be of the form
    folds = []
    for i in 1:(k - 1)
        test_indices = Set(sample(collect(indices_remaining), fold_size))
        train_indices = setdiff(all_indices, test_indices)

        # remove these test indices from the list of indices
        setdiff!(indices_remaining, test_indices)

        # add to folds list
        push!(folds, (collect(train_indices), collect(test_indices)))
    end

    # last fold is just whatever's left
    last_test_indices = indices_remaining
    last_train_indices = setdiff(all_indices, last_test_indices)

    push!(folds, (collect(last_test_indices), collect(last_train_indices)))

    # convert indices to dataframes
    return [(data[pair[1], :], data[pair[2], :]) for pair in folds]
end

folds = get_cross_val_splits(train, 10)

[nrow(pair[2]) for pair in folds]

#-----Fitting Model-----#

# preprocessing data:
X_train = process_data(train, training_info)
y_train = train[!, :Survived]
# defining model
model_formula = @formula(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked)
lr_model = glm(model_formula, X_train, Binomial(), LogitLink())


preds = convert.(Int, round.(Vector{Float64}(predict(lr_model, X_train))))

accuracy = mean(preds .== y_train)

mine1 = Set{Int}([1,2, 3])
mine2 = Set{Int}([2, 3])

setdiff(mine1, mine2)

collect(mine1)
