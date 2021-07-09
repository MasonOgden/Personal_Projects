#%% Packages
#---------------------------------------------------------#
using CSV, DataFrames, Statistics, MLJ, Plots, Random, Dates
using Impute: srs
using StatsPlots: groupedbar
using StatsPlots: @df
using Pipe: @pipe
#---------------------------------------------------------#

## Data

#%% Reading in the Data
#---------------------------------------------------------#
user = "mogde"
cd("C:/Users/$user/Desktop/Git Repositories/Personal_Projects/Kaggle/AnimalShelter")
train_raw = "datasets/train.csv" |> CSV.File |> DataFrame
test_raw = "datasets/test.csv" |> CSV.File |> DataFrame
#---------------------------------------------------------#

#%% Taking a small sample
#---------------------------------------------------------#
num_in_sample = 5000
train_raw_small = train_raw[shuffle(1:nrow(train_raw))[1:num_in_sample], :]
#---------------------------------------------------------#

### Data Cleaning

#%% Examining missing Values
#---------------------------------------------------------#
function num_missing_per_column(dataframe)
    num_missing(colname) = dataframe[:, colname] .|> ismissing |> sum

    @pipe DataFrame(column = names(dataframe),
                    num_missing = dataframe |> names .|> num_missing
                    ) |>
                filter(row -> row.num_missing > 0, _)
end

train_raw_small |> num_missing_per_column
#---------------------------------------------------------#

#%% Data Cleaning Functions
#---------------------------------------------------------#
function rename_cols(dataframe, is_test)
    if is_test
        @pipe dataframe |> 
            rename(_, :ID => :animal_id,
                   :Name => :name,
                   :DateTime => :outcome_datetime,
                   :AnimalType => :animal_type,
                   :SexuponOutcome => :sex,
                   :AgeuponOutcome => :age,
                   :Breed => :breed,
                   :Color => :color)
    else
        @pipe dataframe |> 
            rename(_, :AnimalID => :animal_id,
                   :Name => :name,
                   :DateTime => :outcome_datetime,
                   :OutcomeType => :outcome,
                   :OutcomeSubtype => :outcome_subtype,
                   :AnimalType => :animal_type,
                   :SexuponOutcome => :sex,
                   :AgeuponOutcome => :age,
                   :Breed => :breed,
                   :Color => :color)
    end
end

function age_to_years(age_string)
    if ismissing(age_string)
        return age_string
    end

    extracted_int = parse(Int64, match(r"\d+", age_string).match)

    if occursin(r"year", age_string)
        extracted_int
    elseif occursin(r"month", age_string)
        extracted_int / 12
    elseif occursin(r"week", age_string)
        extracted_int / 52.1429
    else # day
        extracted_int / 365
    end
end

function collapse_color(color_string)
    if occursin(r"Tabby", color_string)
        "Tabby"
    elseif occursin(r"Point", color_string)
        "Point"
    elseif occursin(r"Brindle", color_string)
        "Brindle"
    elseif occursin(r"Calico", color_string)
        "Calico"
    elseif occursin(r"Merle", color_string)
        "Merle"
    elseif occursin(r"Tricolor", color_string)
        "Tricolor"
    elseif occursin(r"Sable", color_string)
        "Sable"
    elseif occursin(r"Blue Tick", color_string) | occursin(r"Blue Cream", color_string)
        "Blue"
    elseif occursin(r"Black Smoke", color_string)
        "Black"
    elseif occursin(r"/", color_string)
        "Multiple Colors"
    elseif occursin(r"Tortie", color_string) | occursin(r"Torbie", color_string)
        "Tortie/Torbie"
    else
        color_string
    end
end

is_day(test_date, test_month, test_day) = month(test_date) == test_month && day(test_date) == test_day

function mutate_data(dataframe, is_test)
    dateformat = DateFormat("y-m-d H:M:S")

    # make copy of dataframe so I don't modify the original
    out_df  = copy(dataframe)
    # convert age to a numeric variable
    out_df[!, :age] = convert.(Union{Float64, Missing}, age_to_years.(out_df[!, :age]))
    # convert string vars to categorical
    out_df[!, :animal_type] = categorical(out_df[!, :animal_type])
    out_df[!, :sex] = categorical(out_df[!, :sex])
    # reduce number of unique animal colors
    out_df[!, :color] = categorical(collapse_color.(out_df[!, :color]))
    # get month from datetime
    # first, convert to datetime
    out_df[!, :outcome_datetime] = DateTime.(out_df[!, :outcome_datetime], dateformat)

    out_df[!, :month] = categorical(month.(out_df[!, :outcome_datetime]))
    # get year from datetime
    out_df[!, :year] = categorical(year.(out_df[!, :outcome_datetime]))

    # add indicator variables for if date is a holiday
    out_df[!, :is_christmaseve] = convert.(Int64, is_day.(out_df[!, :outcome_datetime], 12, 24))
    out_df[!, :is_nyd] = convert.(Int64, is_day.(out_df[!, :outcome_datetime], 1, 1))
    out_df[!, :is_christmas] = convert.(Int64, is_day.(out_df[!, :outcome_datetime], 12, 25))

    if !is_test
        out_df[!, :outcome] = categorical(out_df[!, :outcome])
        select!(out_df, Not(:outcome_subtype))
    end

    return select(out_df, Not([:outcome_datetime, :breed, :name, :animal_id]))
end

drop_missing(dataframe) = dataframe[completecases(dataframe), :]

function finalize_types(dataframe)
    out_df = copy(dataframe)
    out_df[!, :animal_type] = convert(CategoricalArray, out_df[!, :animal_type])
    out_df[!, :sex] = convert(CategoricalArray, convert.(String, out_df[!, :sex]))
    out_df[!, :age] = convert(Vector{Float64}, out_df[!, :age])
    out_df[!, :color] = convert(CategoricalArray, out_df[!, :color])
    out_df[!, :month] = convert(CategoricalArray, out_df[!, :month])
    out_df[!, :year] = convert(CategoricalArray, out_df[!, :year])
    return out_df
end

clean_data(dataframe, is_test_data) = @pipe dataframe |>
                                                rename_cols(_, is_test_data) |> 
                                                mutate_data(_, is_test_data) |>
                                                drop_missing |> 
                                                finalize_types
#---------------------------------------------------------#

#%% Applying Data Cleaning
#---------------------------------------------------------#
train = clean_data(train_raw_small, false)
test = clean_data(test_raw, true)

train |> schema
#---------------------------------------------------------#

### Data Preprocessing

#%% Data Preprocessing Function
#---------------------------------------------------------#
preprocess_data(dataframe, training_info) = @pipe dataframe |>
 	MLJ.transform(training_info["standardizer_fit"], _) |> 
    MLJ.transform(training_info["dummifier_fit"], _)
#---------------------------------------------------------#

#%% Applying Preprocessing
#---------------------------------------------------------#
# split training data into features and labels
X_train = select(train, Not(:outcome))
y_train = train[!, :outcome]

#X_test = test


hot = OneHotEncoder(ordered_factor = false, drop_last = true)
hot_fit = MLJ.fit!(machine(hot, X_train))

stand = Standardizer(count = true)
stand_fit = MLJ.fit!(machine(stand, X_train))

# save training information, preprocess features
X_training_info = Dict("dummifier_fit" => hot_fit,
                       "standardizer_fit" => stand_fit)

X_train_proc = preprocess_data(X_train, X_training_info)
#X_test_proc = preprocess_data(X_test, X_training_info)

X_train_proc |> schema
#---------------------------------------------------------#

## Exploratory Data Analysis

#%% Getting all the data clean
#---------------------------------------------------------#
train_all = clean_data(train_raw, false)
train_all |> size
#---------------------------------------------------------#

#%% Outcomes
#---------------------------------------------------------#
outcome_counts = @pipe train_all |> 
		groupby(_, :outcome) |> 
		combine(_, nrow) |> 
		sort(_, :nrow, rev = true)

outcome_counts[!, :outcome] = convert(Vector{String}, outcome_counts[!, :outcome])

@df outcome_counts bar(
    :outcome,
    :nrow,
    legend = false,
    xlabel = "Outcome",
    ylabel = "Count",
    title = "Distribution of Animal Outcomes in the Training Data")
#---------------------------------------------------------#

#%% Animal Types
#---------------------------------------------------------#
animal_type_props = @pipe train_all |> 
		groupby(_, :animal_type) |> 
		combine(_, nrow) |> 
		sort(_, :nrow, rev = true)

animal_type_props[!, :animal_type] = convert(Vector{String}, animal_type_props[!, :animal_type])

animal_type_props[!, :prop] = animal_type_props[!, :nrow] ./ sum(animal_type_props[!, :nrow])

animal_type_props

@df animal_type_props bar(
    :animal_type,
    :prop,
    legend = false,
    xlabel = "Animal Type",
    ylabel = "Frequency",
    title = "Proportion of Cats and Dogs in the Training Data")
#---------------------------------------------------------#

#%% Sex and Status
#---------------------------------------------------------#
function extract_sex(sex_value) 
    if occursin(r"Male", sex_value)
        "Male"
    elseif occursin("Female", sex_value)
        "Female"
    else
        sex_value
    end
end

function extract_status(sex_value)
    if occursin(r"Neutered", sex_value) | occursin(r"Spayed", sex_value)
        "Spayed/Neutered"
    elseif occursin("Intact", sex_value) 
        "Intact"
    else
        sex_value
    end
end

train_all[!, :sex_bin] = extract_sex.(train_all[!, :sex])
train_all[!, :status] = extract_status.(train_all[!, :sex])

not_intact = [9779, 8819, 0]
intact = [3519, 3504, 0]
unknown = [0, 0, 1089]


groupedbar([not_intact intact unknown],
    bar_position = :stack,
    bar_width = 0.7,
    xticks = (1:3, ["Male", "Female", "Unknown"]),
    label = ["Spayed/Neutered" "Intact" "Unknown"],
    title = "Sex and Status of 26,728 Animals in the Training Data",
    xlabel = "Sex",
    ylabel = "Count"
)
#---------------------------------------------------------#

#%% Age
#---------------------------------------------------------#
@df train_all histogram(
    :age,
    bins = 30,
    xlabel = "Age (years)",
    ylabel = "Count",
    title = "Distribution of Animal Age in Training Data",
    legend = false
)
#---------------------------------------------------------#

#%% Color
#---------------------------------------------------------#
color_counts = @pipe train_all |> 
    groupby(_, :color) |> 
    combine(_, nrow) |> 
    sort(_, :nrow, rev = true)

color_counts[!, :color] = convert(Vector{String}, color_counts[!, :color])

@df color_counts bar(
    :color,
    :nrow,
    orientation = :h,
    yflip = true,
    legend = false,
    xlabel = "Count",
    ylabel = "Animal Coloring",
    title = "Animal Coloring of Animals in the Training Data"
)
#---------------------------------------------------------#

#%% Month
#---------------------------------------------------------#
month_counts = @pipe train_all |> 
    groupby(_, :month) |> 
    combine(_, nrow)

month_counts[!, :month] = convert(Vector{Float64}, month_counts[!, :month])

@df month_counts bar(
    :month,
    :nrow,
    legend = false,
    xlabel = "Month",
    ylabel = "Count",
    title = "Distribution of Animal Outcome Occurrences by Month")
#---------------------------------------------------------#

#%% Year
#---------------------------------------------------------#
year_counts = @pipe train_all |> 
    groupby(_, :year) |> 
    combine(_, nrow)

year_counts[!, :year] = convert(Vector{Float64}, year_counts[!, :year])

@df year_counts bar(
    :year,
    :nrow,
    legend = false,
    xlabel = "Year",
    ylabel = "Count",
    title = "Distribution of Animal Outcome Occurrences by Year")
#---------------------------------------------------------#

## Modeling

### Baseline Results

#%% Getting Data into the correct format
#---------------------------------------------------------#
X = X_train_proc |> Matrix |> MLJ.table
y = y_train

X_train |> size
#---------------------------------------------------------#

#%% Defining Models
#---------------------------------------------------------#
model_names = [
    "adaboost",
    "bagging_clf",
    "bayesian_qda",
    "decision_tree",
    "extra_trees",
    "knn",
    "random_forest",
    "xgb"
]

all_models = [
    (@load AdaBoostClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load BaggingClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load BayesianQDA pkg = ScikitLearn verbosity = 0)(),
    (@load DecisionTreeClassifier pkg = DecisionTree verbosity = 0)(),
    (@load ExtraTreesClassifier pkg = ScikitLearn verbosity = 0)(),
    (@load KNNClassifier pkg = NearestNeighborModels verbosity = 0)(),
    (@load RandomForestClassifier pkg = DecisionTree verbosity = 0)(),
    (@load XGBoostClassifier pkg = XGBoost verbosity = 0)()
]
#---------------------------------------------------------#

#%% Getting cross-validated accuracy for each model
#---------------------------------------------------------#
baseline_results = @pipe DataFrame(
	model_name = model_names,
	cv_acc = [MLJ.evaluate(clf, X, y, resampling=CV(shuffle=true, nfolds=10),
					   measure=accuracy, verbosity=0,
					   operation=predict_mode).measurement[1]
			  for clf ∈ all_models]
) |> sort(_, :cv_acc, rev = true)
#---------------------------------------------------------#

#%% Plotting baseline Results
#---------------------------------------------------------#
@df baseline_results bar(
	:model_name,
	:cv_acc,
	orientation = :h,
	yflip = true,
	legend = false,
	xlabel = "Cross-Validated Accuracy",
	title = "Comparing Baseline Models"
)
#---------------------------------------------------------#

### Hyperparameter Tuning

#%% KNN
#---------------------------------------------------------#
knn_mod = (@load KNNClassifier pkg = NearestNeighborModels verbosity = 0)()

nn_range = range(knn_mod, :K, lower = 2, upper = 50)

tuned_knn_mach = @pipe knn_mod |> 
                    TunedModel(model=_,
                               tuning=Grid(resolution=49),
                               resampling=CV(shuffle=true, nfolds=5),
                               ranges=[nn_range],
                               operation=predict_mode,
                               measure=accuracy) |> 
                    machine(_, X, y)

MLJ.fit!(tuned_knn_mach)

knn_params = tuned_knn_mach |> fitted_params
knn_report = tuned_knn_mach |> report

knn_params.best_model
knn_report.best_history_entry #63.62 accuracy

accs = [knn_report.history[i].measurement[1] for i in 1:size(knn_report.history, 1)]

final_knn_clf = (@load KNNClassifier pkg = NearestNeighborModels verbosity = 0)(
    K = knn_params.best_model.K # 39
)
	
plot(
    2:(size(accs, 1) + 1), 
    accs, 
    legend = false, 
    xlabel = "K",
    ylabel = "10-Fold CV Accuracy",
    title = "Tuning K by Cross-Validated Accuracy"
)
#---------------------------------------------------------#

#%% AdaBoost
#---------------------------------------------------------#
ada = (@load AdaBoostClassifier pkg = ScikitLearn verbosity = 0)()
	
n_estimators_range = range(ada, :n_estimators, lower = 25, upper = 200)
learning_rate_range = range(ada, :learning_rate, lower = 0.5, upper = 2)

tuned_ada_mach = @pipe ada |> 
                    TunedModel(model=_,
                               tuning=Grid(resolution=11),
                               resampling=CV(shuffle=true, nfolds=5),
                               ranges=[n_estimators_range, learning_rate_range],
                               operation=predict_mode,
                               measure=accuracy) |>
                    machine(_, X, y)

MLJ.fit!(tuned_ada_mach)

ada_params = tuned_ada_mach |> fitted_params
ada_report = tuned_ada_mach |> report

ada_params.best_model
ada_report.best_history_entry # 60.40% accuracy

final_ada_clf = (@load AdaBoostClassifier pkg = ScikitLearn verbosity = 0)(
    n_estimators = ada_params.best_model.n_estimators, # 42
    learning_rate = ada_params.best_model.learning_rate # 0.5
)
#---------------------------------------------------------#

#%% XGBoost
#---------------------------------------------------------#
xgb1 = (@load XGBoostClassifier pkg = XGBoost verbosity = 0)()

max_depth_range = range(xgb1, :max_depth, lower = 3, upper = 10)
γ_range = range(xgb1, :gamma, lower = 0.0, upper = 0.2)

tuned_xgb1_mach = @pipe xgb1 |> 
                     TunedModel(model=_,
                             tuning=Grid(resolution=11),
                             resampling=CV(shuffle=true, nfolds=5),
                             ranges=[max_depth_range, γ_range],
                             operation=predict_mode,
                             measure=accuracy) |>
                     machine(_, X, y)

MLJ.fit!(tuned_xgb1_mach)


xgb1_params = tuned_xgb1_mach |> fitted_params
xgb1_report = tuned_xgb1_mach |> report

xgb1_params.best_model
xgb1_report.best_history_entry # 63.04% accuracy

#---------------------------------------------------------#
xgb1_params.best_model.max_depth

xgb2 = (@load XGBoostClassifier pkg = XGBoost verbosity = 0)(
    max_depth = xgb1_params.best_model.max_depth, # 3
    gamma = xgb1_params.best_model.gamma # 0.08
)

num_round_range = range(xgb2, :num_round, lower = 75, upper = 150)
booster_range = range(xgb2, :booster, values = ["gbtree", "gblinear", "dart"])
η_range = range(xgb2, :eta, lower = 0.1, upper = 0.7)

tuned_xgb2_mach = @pipe xgb2 |> 
                     TunedModel(model=_,
                                tuning=Grid(resolution=5),
                                resampling=CV(shuffle=true, nfolds=5),
                                ranges=[num_round_range, booster_range, η_range],
                                operation=predict_mode,
                                measure=accuracy) |>
                     machine(_, X, y)

MLJ.fit!(tuned_xgb2_mach)


xgb2_params = tuned_xgb2_mach |> fitted_params
xgb2_report = tuned_xgb2_mach |> report

xgb2_params.best_model
xgb2_report.best_history_entry #63.07% accuracy

final_xgb = (@load XGBoostClassifier pkg = XGBoost verbosity = 0)(
    max_depth = xgb1_params.best_model.max_depth, # 3
    gamma = xgb1_params.best_model.gamma, # 0.08,
    num_round = xgb2_params.best_model.num_round, # 75
    booster = xgb2_params.best_model.booster, # "gbtree"
    eta = xgb2_params.best_model.eta # 0.55
)

#%% Bagging Classifier

Λ = (@load BaggingClassifier pkg = ScikitLearn verbosity = 0)()

n_estimators_range = range(Λ, :n_estimators, lower = 5, upper = 20)
max_samples_range = range(Λ, :max_samples, lower = 0.5, upper = 1.0)
max_features_range = range(Λ, :max_features, lower = 0.5, upper = 1.0)

tuned_Λ_mach = @pipe Λ |> 
    TunedModel(model = _,
               tuning = Grid(resolution = 5),
               resampling = CV(shuffle = true, nfolds = 5),
               ranges = [n_estimators_range, max_samples_range, max_features_range],
               operation = predict_mode,
               measure = accuracy) |> 
    machine(_, X, y)

MLJ.fit!(tuned_Λ_mach)

Λ_params = tuned_Λ_mach |> fitted_params
Λ_report = tuned_Λ_mach |> report

Λ_params.best_model
Λ_report.best_history_entry # 61.22% accuracy

final_bagged_clf = (@load BaggingClassifier pkg = ScikitLearn verbosity = 0)(
    n_estimators = Λ_params.best_model.n_estimators, # 20
    max_samples = Λ_params.best_model.max_samples, # 0.625
    max_features = Λ_params.best_model.max_features # 0.5
)

#%% Random Forest

rf1 = (@load RandomForestClassifier pkg = DecisionTree verbosity = 0)()

max_depth_range = range(rf1, :max_depth, values = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
min_samples_split_range = range(rf1, :min_samples_split, lower = 2, upper = 10)

tuned_rf1_mach = @pipe rf1 |> 
    TunedModel(model = _,
               tuning = Grid(resolution = 11),
               resampling = CV(shuffle = true, nfolds = 5),
               ranges = [max_depth_range, min_samples_split_range],
               operation = predict_mode,
               measure = accuracy) |> 
    machine(_, X, y)

MLJ.fit!(tuned_rf1_mach)

rf1_params = tuned_rf1_mach |> fitted_params
rf1_report = tuned_rf1_mach |> report

rf1_params.best_model
rf1_report.best_history_entry # 61.32% accuracy

rf2 = (@load RandomForestClassifier pkg = DecisionTree verbosity = 0)(
    max_depth = rf1_params.best_model.max_depth, # 10
    min_samples_split = rf1_params.best_model.min_samples_split, # 4
)

n_subfeatures_range = range(rf2, :n_subfeatures, values = [-1, 10, 20, 30, 40, 49])
n_trees_range = range(rf2, :n_trees, lower = 5, upper = 100)
sampling_fraction_range = range(rf2, :sampling_fraction, lower = 0.5, upper = 0.9)

tuned_rf2_mach = @pipe rf2 |> 
    TunedModel(model = _,
               tuning = Grid(resolution = 5),
               resampling = CV(shuffle = true, nfolds = 5),
               ranges = [n_subfeatures_range, n_trees_range, sampling_fraction_range],
               operation = predict_mode,
               measure = accuracy) |> 
    machine(_, X, y)

MLJ.fit!(tuned_rf2_mach)

rf2_params = tuned_rf2_mach |> fitted_params
rf2_report = tuned_rf2_mach |> report

rf2_params.best_model
rf2_report.best_history_entry # 63.84% accuracy

final_rf = (@load RandomForestClassifier pkg = DecisionTree verbosity = 0)(
    max_depth = rf1_params.best_model.max_depth, # 10
    min_samples_split = rf1_params.best_model.min_samples_split, # 4
    n_subfeatures = rf2_params.best_model.n_subfeatures, # 20
    n_trees = rf2_params.best_model.n_trees, # 100
    sampling_fraction = rf2_params.best_model.sampling_fraction # 0.8
)

#%% ExtraTrees

xt1 = (@load ExtraTreesClassifier pkg = ScikitLearn verbosity = 0)()

max_depth_range = range(Int, :max_depth, lower = 2, upper = 15)

tuned_xt1_mach = @pipe xt1 |> 
    TunedModel(model = _,
               tuning = Grid(resolution = 11),
               resampling = CV(shuffle = true, nfolds = 5),
               ranges = [max_depth_range, min_samples_split_range],
               operation = predict_mode,
               measure = accuracy) |> 
    machine(_, X, y)

MLJ.fit!(tuned_xt1_mach)

xt1_params = tuned_xt1_mach |> fitted_params
xt1_report = tuned_xt1_mach |> report

xt1_params.best_model
xt1_report.best_history_entry # 62.60% accuracy



xt2 = (@load ExtraTreesClassifier pkg = ScikitLearn verbosity = 0)(
    max_depth = xt1_params.best_model.max_depth, # 14
    min_samples_split = xt1_params.best_model.min_samples_split # 4
)

n_estimators_range = range(xt2, :n_estimators, lower = 50, upper = 150)
max_features_range = range(xt2, :max_features, values = ["sqrt", "log2"])

tuned_xt2_mach = @pipe xt2 |> 
    TunedModel(model = _,
               tuning = Grid(resolution = 50),
               resampling = CV(shuffle = true, nfolds = 5),
               ranges = [n_estimators_range, max_features_range],
               operation = predict_mode,
               measure = accuracy) |> 
    machine(_, X, y)

MLJ.fit!(tuned_xt2_mach)

xt2_params = tuned_xt2_mach |> fitted_params
xt2_report = tuned_xt2_mach |> report

xt2_params.best_model
xt2_report.best_history_entry # 62.50% accuracy

final_xt = (@load ExtraTreesClassifier pkg = ScikitLearn verbosity = 0)(
    max_depth = xt1_params.best_model.max_depth, # 11
    min_samples_split = xt1_params.best_model.min_samples_split, # 5
    n_estimators = xt2_params.best_model.n_estimators, # 148
    max_features = xt2_params.best_model.max_features, # "sqrt"
)

#%% Comparing Tuned Models

tuned_names = [
    "KNN",
    "AdaBoost",
    "XGBoost",
    "Bagging CLF",
    "Random Forest",
    "ExtraTrees"
]

tuned_models = [
    final_knn_clf,
    final_ada_clf, 
    final_xgb,
    final_bagged_clf,
    final_rf,
    final_xt
]

tuned_results = @pipe DataFrame(
	model_name = tuned_names,
	cv_acc = [MLJ.evaluate(clf, X, y, resampling=CV(shuffle=true, nfolds=10),
					       measure=accuracy, verbosity=0,
					       operation=predict_mode).measurement[1]
			  for clf ∈ tuned_models]
) |> sort(_, :cv_acc, rev = true)

@df tuned_results bar(
	:model_name,
	:cv_acc,
	orientation = :h,
	yflip = true,
	legend = false,
	xlabel = "Cross-Validated Accuracy",
	title = "Comparing Tuned Models"
)

# best is random forest with 63.62% accuracy