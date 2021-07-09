#%% Packages
using CSV, DataFrames, Statistics, MLJ, Plots
using Impute: srs
using StatsPlots: @df
using Pipe: @pipe

#%% Data
user = "mogde"

cd("C:/Users/$user/Desktop/Git Repositories/Personal_Projects/Kaggle/Titanic")
train_raw = "datasets/train.csv" |> CSV.File |> DataFrame
test_raw = "datasets/test.csv" |> CSV.File |> DataFrame

#%% Data Cleaning
# checking how many missing values there are and in which columns

function num_missing_per_column(dataframe)
    num_missing(colname) = dataframe[:, colname] .|> ismissing |> sum

    @pipe DataFrame(column = names(dataframe),
                    num_missing = dataframe |> names .|> num_missing
                    ) |>
                filter(row -> row.num_missing > 0, _)
end

train_raw |> num_missing_per_column

# removes unnecessary columns, converts to correct types
function clean_data(in_data, is_test)
    if is_test
        @pipe in_data |> 
            transform!(_, [:Pclass, :Sex, :Embarked] .=> categorical) |>
            select(_, Not([:Pclass, :Sex, :Embarked]), 
                :Pclass_categorical => :passenger_class,
                :Sex_categorical => :sex,
                :Embarked_categorical => :embarked_location,
                :Age => :age,
                :SibSp => :num_siblings,
                :Parch => :parch,
                :Fare => :fare) |>
            select(_, [:age, :num_siblings, :parch, :fare,
                       :passenger_class, :sex, :embarked_location])
    else
        @pipe in_data |> 
            transform!(_, [:Survived, :Pclass, :Sex, :Embarked] .=> categorical) |>
            select(_, Not([:Pclass, :Sex, :Embarked]), 
                :Survived_categorical => :survived,
                :Pclass_categorical => :passenger_class,
                :Sex_categorical => :sex,
                :Embarked_categorical => :embarked_location,
                :Age => :age,
                :SibSp => :num_siblings,
                :Parch => :parch,
                :Fare => :fare) |>
            select(_, [:survived, :age, :num_siblings, :parch, :fare,
                   :passenger_class, :sex, :embarked_location])

    end
end

train = @pipe train_raw |> clean_data(_, false)
test = @pipe test_raw |> clean_data(_, true)

train |> schema

#%% Preprocessing Data

X_train = @pipe train |> select(_, Not(:survived)) |> Matrix |> MLJ.table
y_train = train[!, :survived]

# define what type of imputation, encoding, and standardization I want
training_info = Dict(
    "imputer" => srs,
    "dummifier" => MLJ.OneHotEncoder(ordered_factor = false, drop_last = true),
    "standardizer" => MLJ.Standardizer(count = true)
)

function preprocess_data(X, training_info)
    # impute with SRS
    X_imputed = training_info["imputer"](X) |> DataFrame
    X_imputed[!, :x1] = convert.(Float64, X_imputed[!, :x1])
    X_imputed[!, :x2] = convert.(Int64, X_imputed[!, :x2])
    X_imputed[!, :x3] = convert.(Int64, X_imputed[!, :x3])
    X_imputed[!, :x4] = convert.(Float64, X_imputed[!, :x4])
    X_imputed[!, :x5] = convert.(Int64, X_imputed[!, :x5])
    X_imputed[!, :x6] = categorical(convert(Vector{String}, X_imputed[:, :x6]))
    X_imputed[!, :x7] = categorical(convert(Vector{String}, X_imputed[:, :x7]))

    # fit standardizer to imputed data
    standardizer_fit = MLJ.fit!(machine(training_info["standardizer"], X_imputed))
    # fit dummifier to imputed data
    dummifier_fit = MLJ.fit!(machine(training_info["dummifier"], X_imputed))

    # apply standardizer and dummifier
    @pipe X_imputed |> 
        MLJ.transform(standardizer_fit, _) |> 
        MLJ.transform(dummifier_fit, _) |> 
        Matrix |> 
        MLJ.table
end

X = preprocess_data(X_train, training_info)
y = y_train

#%% Getting Baseline Model Results

model_names = [
    #"adaboost",
	"bagging_clf",
	"bayesian_qda",
	"decision_tree",
	"extra_trees",
	"knn",
	"kernel_perceptron",
    "linear_binary",
    "logistic",
	"random_forest",
	"xgb"
]

all_models = [
    #(@load AdaBoostClassifier pkg = ScikitLearn),
	(@load BaggingClassifier pkg = ScikitLearn)(),
	(@load BayesianQDA pkg = ScikitLearn)(),
	(@load DecisionTreeClassifier pkg = DecisionTree)(),
	(@load ExtraTreesClassifier pkg = ScikitLearn)(),
	(@load KNNClassifier pkg = NearestNeighborModels)(),
	(@load KernelPerceptronClassifier pkg = BetaML)(),
    (@load LinearBinaryClassifier pkg = GLM)(),
    (@load LogisticClassifier pkg = MLJLinearModels)(),
	(@load RandomForestClassifier pkg = DecisionTree)(),
	(@load XGBoostClassifier pkg = XGBoost)()
]

baseline_results = @pipe DataFrame(
	model_name = model_names,
	cv_acc = [MLJ.evaluate(clf, X, y, resampling=CV(shuffle=true, nfolds=5),
					   measure=accuracy, verbosity=0,
					   operation=predict_mode).measurement[1]
			  for clf ∈ all_models]
) |> sort(_, :cv_acc, rev = true)

@df baseline_results bar(
	:model_name,
	:cv_acc,
	orientation = :h,
	yflip = true,
	legend = false,
	xlabel = "Cross-Validated Accuracy",
	title = "Comparing Baseline Models"
)

#%% Tuning Best Models

# ExtraTrees

xt = (@load ExtraTreesClassifier pkg = ScikitLearn)()

trees_range = range(xt, :n_estimators, lower = 50, upper = 150)
min_samples_split_range = range(xt, :min_samples_split, lower = 2, upper = 10)
min_samples_leaf_range = range(xt, :min_samples_leaf, lower = 1, upper = 7)

tuned_xt = TunedModel(model=xt,
					  tuning=Grid(resolution=5),
					  resampling=CV(shuffle=true, nfolds=5),
					  ranges=[trees_range, min_samples_split_range, min_samples_leaf_range],
					  operation=predict_mode,
					  measure=accuracy)

tuned_xt_mach = machine(tuned_xt, X, y)

MLJ.fit!(tuned_xt_mach)

xt_params = tuned_xt_mach |> fitted_params
xt_report = tuned_xt_mach |> report

xt_params.best_model
xt_report.best_history_entry # 83.16% accuracy


# Bagging Classifier

ψ = (@load BaggingClassifier pkg = ScikitLearn)()

n_estimators_range = range(ψ, :n_estimators, lower = 10, upper = 100)
max_samples_range = range(ψ, :max_samples, lower = 0.5, upper = 1.0)
max_features_range = range(ψ, :max_features, lower = 0.5, upper = 1.0)

tuned_ψ = TunedModel(model=ψ,
					  tuning=Grid(resolution=5),
					  resampling=CV(shuffle=true, nfolds=5),
					  ranges=[n_estimators_range, max_samples_range, max_features_range],
					  operation=predict_mode,
					  measure=accuracy)

tuned_ψ_mach = machine(tuned_ψ, X, y)

MLJ.fit!(tuned_ψ_mach)

ψ_params = tuned_ψ_mach |> fitted_params
ψ_report = tuned_ψ_mach |> report

ψ_params.best_model
ψ_report.best_history_entry # 83.50% accuracy

# XGBoost

xgb = (@load XGBoostClassifier pkg = XGBoost)()

# tuning greek letter parameters

η_range = range(xgb, :eta, lower = 0.1, upper = 0.5)
γ_range = range(xgb, :gamma, lower = 0.01, upper = 0.1)
λ_range = range(xgb, :lambda, lower = 0.5, upper = 1.0)
α_range = range(xgb, :alpha, lower = 0, upper = 0.5)

tuned_xgb = TunedModel(model=xgb,
					   tuning=Grid(resolution=4),
					   resampling=CV(shuffle=true, nfolds=5),
					   ranges=[η_range, γ_range, λ_range, α_range],
					   operation=predict_mode,
					   measure=accuracy)

tuned_xgb_mach = machine(tuned_xgb, X, y)

MLJ.fit!(tuned_xgb_mach)

xgb_params = tuned_xgb_mach |> fitted_params
xgb_report = tuned_xgb_mach |> report

xgb_params.best_model
xgb_report.best_history_entry # 82.06% accuracy

# tuning other parameters

xgb2 = (@load XGBoostClassifier pkg = XGBoost)(eta = 0.23333333333333334, gamma = 0.01, lambda = 0.5, alpha = 0.5)

max_depth_range = range(xgb2, :max_depth, lower = 2, upper = 10)
num_round_range = range(xgb2, :num_round, lower = 50, upper = 200)
colsample_bylevel_range = range(xgb2, :colsample_bylevel, lower = 0.7, upper = 1.0)


tuned_xgb2 = TunedModel(model=xgb2,
					    tuning=Grid(resolution=5),
					    resampling=CV(shuffle=true, nfolds=5),
					    ranges=[max_depth_range, num_round_range, colsample_bylevel_range],
					    operation=predict_mode,
					    measure=accuracy)

tuned_xgb2_mach = machine(tuned_xgb2, X, y)

MLJ.fit!(tuned_xgb2_mach)

xgb2_params = tuned_xgb2_mach |> fitted_params
xgb2_report = tuned_xgb2_mach |> report

xgb2_params.best_model
xgb2_report.best_history_entry # 82.94% accuracy