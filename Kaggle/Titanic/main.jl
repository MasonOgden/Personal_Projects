#-----Packages-----#
using CSV, DataFrames, StatsModels, GLM, Statistics, StatsBase, ScikitLearn, Lathe, Missings

#-----Data-----#
cd("C:/Users/Mason/Desktop/Git Repositories/Personal_Projects/Kaggle/Titanic")
train = DataFrame(CSV.File("datasets/train.csv"))
test = DataFrame(CSV.File("datasets/test.csv"))

DataFrame(
    col_name = train |> names,
    col_type = eltype.(eachcol(train))
)

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
    transform!(dataframe, [:Pclass, :Sex, :Embarked] .=> categorical, renamecols=false)
    transform!(dataframe, [:SibSp, :Parch] .=> float, renamecols=false)
    #transform!(dataframe, :Pclass => categorical, renamecols=false)
    #transform!(dataframe, :Sex => categorical, renamecols=false)
    #transform!(dataframe, :Embarked => categorical, renamecols=false)
    #dataframe[!, :SibSp] = convert.(AbstractFloat, dataframe[:, :SibSp])
    #dataframe[!, :Parch] = convert.(AbstractFloat, dataframe[:, :Parch])

    # impute columns with missing values
    dataframe[:, :Age] = coalesce.(dataframe[:, :Age], training_info["age_median"])
    dataframe[:, :Embarked] = coalesce.(dataframe[:, :Embarked], training_info["embarked_mode"])

    # scaling numeric
    dataframe[:, :Age] = training_info["age_scaler"].predict(dataframe[:, :Age]) |> skipmissing |> collect
    dataframe[:, :SibSp] = training_info["sibsp_scaler"].predict(dataframe[:, :SibSp])
    dataframe[:, :Parch] = training_info["parch_scaler"].predict(dataframe[:, :Parch])
    dataframe[:, :Fare] = training_info["fare_scaler"].predict(dataframe[:, :Fare])

    # creating dummy variables, drop originals
    dummifier = Lathe.preprocess.OneHotEncoder()
    dummifier.predict(dataframe, :Pclass)
    dummifier.predict(dataframe, :Sex)
    dummifier.predict(dataframe, :Embarked)
    select!(dataframe, Not([:Pclass, :Sex, :Embarked]))

    # now convert boolean columns to 0/1 columns
    transform!(dataframe, ["1", "2", "3", "S", "C", "Q", "male", "female"] .=> float, renamecols=false)

    return dataframe
end


training_info = Dict(
    "age_median" => train[:, :Age] |> skipmissing |> median,
    "embarked_mode" => train[:, :Embarked] |> skipmissing |> mode |> string,
    "age_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train[:, :Age], train[:, :Age] |> skipmissing |> median)),
    "sibsp_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train[:, :SibSp], train[:, :SibSp] |> skipmissing |> median)),
    "parch_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train[:, :Parch], train[:, :Parch] |> skipmissing |> median)),
    "fare_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train[:, :Fare], train[:, :Fare] |> skipmissing |> median))
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
        test_indices = Set(sample(collect(indices_remaining), fold_size, replace=false))
        train_indices = setdiff(all_indices, test_indices)

        # remove these test indices from the list of indices
        setdiff!(indices_remaining, test_indices)

        # add to folds list
        push!(folds, (collect(train_indices), collect(test_indices)))
    end

    # last fold is just whatever's left
    last_test_indices = indices_remaining
    last_train_indices = setdiff(all_indices, last_test_indices)

    push!(folds, (collect(last_train_indices), collect(last_test_indices)))

    # convert indices to dataframes
    return [(data[pair[1], :], data[pair[2], :]) for pair in folds]
end

folds = get_cross_val_splits(train, 10)

#-----Fitting Model-----#
model1_cols = [:Pclass]
model2_cols = [:Pclass, :Sex]
model3_cols = [:Pclass, :Sex, :Age]
model4_cols = [:Pclass, :Sex, :Age, :SibSp]
model5_cols = [:Pclass, :Sex, :Age, :SibSp, :Parch]
model6_cols = [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare]
model7_cols = [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked]

model_col_list= [model1_cols, model2_cols, model3_cols, model4_cols, model5_cols, model6_cols, model7_cols]

function cross_validation_metrics(folds, model)

    out_metrics = Dict()

    tp = 0
    fp = 0
    fn = 0
    tn = 0


    for fold in folds
        # extract training and testing data from the fold
        train_data = fold[1]
        test_data = fold[2]

        # fit preprocessing on training data:
        training_info = Dict(
            "age_median" => train_data[:, :Age] |> skipmissing |> median,
            "embarked_mode" => train_data[:, :Embarked] |> skipmissing |> mode |> string,
            "age_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train_data[:, :Age], train_data[:, :Age] |> skipmissing |> median)),
            "sibsp_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train_data[:, :SibSp], train_data[:, :SibSp] |> skipmissing |> median)),
            "parch_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train_data[:, :Parch], train_data[:, :Parch] |> skipmissing |> median)),
            "fare_scaler" => Lathe.preprocess.StandardScaler(coalesce.(train_data[:, :Fare], train_data[:, :Fare] |> skipmissing |> median))
        )

        # preprocess data, subset to just chosen features
        X = select(process_data(train_data, training_info), Not(:Survived))
        y = train_data[:, :Survived]
        X_test = select(process_data(test_data, training_info), Not(:Survived))
        y_test = test_data[:, :Survived]

        # fit model on training data
        ScikitLearn.fit!(model, X, y)
        # predict on test data:
        preds = convert.(Int, round.(Vector{Float64}(predict(lr_model, X_test))))

        results_df = DataFrame(
            prediction = preds,
            truth = y_test
        )
        tp += filter(row -> (row[1] == 1 & row[2] == 1), results_df) |> nrow
        fp += filter(row -> (row[1] == 1 & row[2] == 0), results_df) |> nrow
        fn += filter(row -> (row[1] == 0 & row[2] == 0), results_df) |> nrow
        tn += filter(row -> (row[1] == 0 & row[2] == 0), results_df) |> nrow

    end
    out_metrics["tp"] = tp
    out_metrics["fp"] = fp
    out_metrics["fn"] = fn
    out_metrics["tn"] = tn

    out_metrics["acc"] = (tp + tn) / (fp + fn)
    out_metrics["prec"] = tp / (tp + fp)

    recall = tp / (tp + fn)
    out_metrics["recall"] = recall
    out_metrics["tpr"] = recall
    out_metrics["fpr"] = fp / (fp + tn)

    return out_metrics
end

model_names = string.(model_list)
model_results = [cross_validation_metrics(folds, model) for model in model_list]

results_df = DataFrame(
    model = model_names,
    accuracy = [metrics_dict["acc"] for metrics_dict in model_results],
    precision = [metrics_dict["prec"] for metrics_dict in model_results],
    recall = [metrics_dict["recall"] for metrics_dict in model_results],
    true_positive_rate = [metrics_dict["tpr"] for metrics_dict in model_results],
    false_positive_rate = [metrics_dict["fpr"] for metrics_dict in model_results],
    )

sort(results_df, :accuracy, rev=true)

#-----ScikitLearn-----#

@sk_import linear_model: LogisticRegression
@sk_import neural_network: MLPClassifier
@sk_import discriminant_analysis: LinearDiscriminantAnalysis
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: (SVC, LinearSVC, NuSVC)
@sk_import tree: DecisionTreeClassifier
@sk_import ensemble: (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)

classifiers = [
    KNeighborsClassifier(3),
    LinearSVC(C=0.025),
    SVC(),
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(fit_intercept=true)
    ]

out = cross_validation_metrics(folds, lr_model)

dummifier = Lathe.preprocess.OneHotEncoder()

dummifier.predict(train, :Sex)

select!(train, Not(:Sex))

X_train = select(process_data(train, training_info), Not(:Survived))
y_train = train[:, :Survived]

DataFrame(
    col_name = X_train |> names,
    col_type = eltype.(eachcol(X_train))
)
X_train[:, :Age] .|> ismissing |> sum

disallowmissing(X_train[:, :Age])

convert(Vector{Float64}, X_train[:, :Age])

ScikitLearn.fit!(lr_model, X_train, y_train)
