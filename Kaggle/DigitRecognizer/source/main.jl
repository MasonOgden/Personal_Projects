#%% Packages
import CSV
using DataFrames, MLJ, CategoricalArrays
using Chain: @chain


#%% Data
dataset_dir = "datasets"

num_rows_to_keep = 2000

y, X = @chain dataset_dir begin
    joinpath("train.csv")
    CSV.File
    DataFrame
    unpack(
        ==(:label),
        !=(:label)
    )
end

chosen_rows = shuffle(1:nrow(X))[1:num_rows_to_keep]

X = @chain X begin
    select(
        Symbol.(names(X)) .=> (x -> float.(x ./ 255)),
        renamecols = false
        )
    _[chosen_rows, :]
end

y = CategoricalArray(y[chosen_rows])

### Initial Modeling

#%% Removing NZV columns

nzv_cols = @chain DataFrame(
    col = names(X),
    num_unique = size.(unique.(eachcol(X)), 1)
) filter(x -> x.num_unique > 1, _) _[!, :col] Symbol.(_)

X_nzv = X[!, nzv_cols]

#%% Defining Models to Try

model_names = [
    "AdaBoostClassifier",
    "AdaBoostStumpClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "KNNClassifier",
    "LDA", 
    "LogisticClassifier",
    "NuSVC", 
    #"PegasosClassifier", 
    "RandomForestClassifier",
    "RidgeClassifier",
    "SGDClassifier",
    "SVMClassifier"
]

model_specs = [
    (@load AdaBoostClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load AdaBoostStumpClassifier pkg = DecisionTree verbosity = 0)(),
    (@load DecisionTreeClassifier pkg = DecisionTree verbosity = 0)(),
    (@load ExtraTreesClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load GradientBoostingClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load KNNClassifier pkg = NearestNeighborModels verbosity = 0)(),
    (@load LDA pkg = MultivariateStats verbosity = 0)(),
    (@load LogisticClassifier pkg = MLJLinearModels verbosity = 0)(),
    (@load NuSVC pkg = LIBSVM verbosity = 0)(),
    #(@load PegasosClassifier pkg = BetaML verbosity = 0)(),
    (@load RandomForestClassifier pkg = DecisionTree verbosity = 0)(),
    (@load RidgeClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load SGDClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load SVMClassifier pkg = ScikitLearn verbosity = 0)()
]

#%% Getting Baseline Results

baseline_results = @chain DataFrame(
	model_name = model_names,
	cv_acc = [
        evaluate(
            clf,
            X_nzv,
            y,
            resampling=CV(shuffle=true, nfolds=5),
			measure=accuracy, verbosity=0,
			operation=predict_mode
            ).measurement[1]
		for clf ∈ model_specs
    ]
) sort(:cv_acc, rev = true)

size(X_nzv)