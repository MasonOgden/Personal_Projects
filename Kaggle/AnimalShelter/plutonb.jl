### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 09a2ce5d-55b8-489c-8f67-5dccdb2d75c9
begin
	#%% Packages
	using CSV, DataFrames, Statistics, MLJ, Plots, Random, Dates
	using Impute: srs
	using StatsPlots: groupedbar
	using StatsPlots: @df
	using Pipe: @pipe
end

# ╔═╡ 3b1b2020-d32b-11eb-1512-b74bcf73efee
md"## Packages"

# ╔═╡ 7b8df1d5-ac46-4fe0-9f8d-7f6059d29c6c
md"## Data"

# ╔═╡ 076bc9f3-8dfe-4307-b169-ac9a558dc411
md"### Reading in the Data"

# ╔═╡ 3c781196-8ad2-4de3-8398-05406b1b3c35
begin
	user = "mogde"
	cd("C:/Users/$user/Desktop/Git Repositories/Personal_Projects/Kaggle/AnimalShelter")
	train_raw = "datasets/train.csv" |> CSV.File |> DataFrame
	test_raw = "datasets/test.csv" |> CSV.File |> DataFrame
	@pipe train_raw |> first(_, 5)
end

# ╔═╡ 67433b97-713e-430c-9be3-4cd5d46dfc21
md"**Taking a small sample**"

# ╔═╡ a81491b1-c799-4381-93fa-064abcdf6315
begin
	num_in_sample = 5000
	train_raw_small = train_raw[shuffle(1:nrow(train_raw))[1:num_in_sample], :]
	@pipe train_raw_small |> first(_, 5)
end

# ╔═╡ 5adca38e-02d2-4967-812d-fcf66b3e563a
md"### Data Cleaning"

# ╔═╡ c7e0cd43-68a2-4458-a504-deaee4004176
md"**Examining Missing Values**"

# ╔═╡ 31afd76c-0610-4131-8123-d56eea703925
begin
	function num_missing_per_column(dataframe)
    num_missing(colname) = dataframe[:, colname] .|> ismissing |> sum

    @pipe DataFrame(column = names(dataframe),
                    num_missing = dataframe |> names .|> num_missing
                    ) |>
                filter(row -> row.num_missing > 0, _)
	end

	train_raw_small |> num_missing_per_column
end

# ╔═╡ 0326b7da-1955-4de0-a94b-f59940c8056b
md"**Data Cleaning Functions**"

# ╔═╡ d24dac74-e9fc-42d4-a781-8de32f3bf44a
begin
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
end

# ╔═╡ 376c0560-8dee-482c-a4db-4748244f27c0
begin
	train = clean_data(train_raw_small, false)
	test = clean_data(test_raw, true)

	train |> schema
end

# ╔═╡ eef1671e-3fd0-4464-854a-c4c342545ad7
md"### Data Preprocessing"

# ╔═╡ 8e0a7fd6-43fe-4c3a-a089-6ca949c94f37
md"**Function to preprocess**"

# ╔═╡ f80df2ec-6cf2-430e-afea-0698935d3c1d
# use fit standardizer to standardize age, then use fit dummifier to dummify categorical variables
preprocess_data(dataframe, training_info) = @pipe dataframe |> 	MLJ.transform(training_info["standardizer_fit"], _) |> 	MLJ.transform(training_info["dummifier_fit"], _)

# ╔═╡ 2cb1c2ac-1e0d-472a-8550-4a5b0dfb08d9
md"**Apply Preprocessing**"

# ╔═╡ a7683b00-4f50-4290-a62e-27d181ff70ad
begin
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
end

# ╔═╡ d347c739-a68e-4881-9a6b-b6c7de8c711c
md"## Exploratory Data Analysis"

# ╔═╡ 9a88c3e5-75b6-4827-9287-8e7bdd76eaa1
md"### Getting all the data clean (not just sample)"

# ╔═╡ 2f143245-529d-48df-8ad0-7c42814128b7
begin
	train_all = clean_data(train_raw, false)
	train_all |> size
end

# ╔═╡ 99fb27dc-18cb-419b-b89e-7d8e7710aa58
md"#### Outcomes"

# ╔═╡ 782159ec-e6c6-40e8-9007-2ff4c29d159d
begin
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
end

# ╔═╡ a8c63877-d106-4cc5-97fd-ec2b16a06a4b
md"#### Animal Type"

# ╔═╡ b8662c8c-4c8a-4326-8246-879aa6dec207
begin
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
end

# ╔═╡ 732e5fd0-c190-48d4-aa9e-82a4184af6cb
md"#### Sex and Status"

# ╔═╡ 5e55c735-4e10-4103-8919-361366ef8edb
begin
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
end

# ╔═╡ 3ceb40dc-23d8-461a-9cec-fc5653719ee3
begin
	train_all[!, :sex_bin] = extract_sex.(train_all[!, :sex])
	train_all[!, :status] = extract_status.(train_all[!, :sex])
end

# ╔═╡ f1145ffe-4e35-4d55-b620-aca82bb3a76d
begin
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
end

# ╔═╡ 5b629a72-b4fd-41aa-be02-eaab904f810e
md"#### Age"

# ╔═╡ c49cb7bd-80d3-456a-818b-380c52c29bb3
@df train_all histogram(
    :age,
    bins = 30,
    xlabel = "Age (years)",
    ylabel = "Count",
    title = "Distribution of Animal Age in Training Data",
    legend = false
)

# ╔═╡ ac82eb5c-898c-4bde-9fd8-92711b016d80
md"#### Color"

# ╔═╡ b6401293-7a4c-46a2-b935-2855cb47f2f4
begin
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
end

# ╔═╡ 0055bf18-af67-4105-98fe-dab8fbdc3e5d
md"#### Month"

# ╔═╡ 7a1101f1-d78c-4300-91e3-f017c451fa98
begin
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
end

# ╔═╡ df238f00-6e4e-4b3c-8075-6a846e9e959c
md"#### Year"

# ╔═╡ 521628e0-3614-44bf-a063-b54f5f373197
begin
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
end

# ╔═╡ f55876b0-9196-4b7e-8f35-8925d15190c6
md"""
## Modeling
"""

# ╔═╡ 04bc5320-6fb2-44ee-a7b8-bcf7191cafa6
md"""
**Get data into correct format**
"""

# ╔═╡ f9664918-12cd-404d-93a4-9e3aa2e81ddf
begin
	X = X_train_proc |> Matrix |> MLJ.table
	y = y_train

	X_train |> size
end

# ╔═╡ 015b282b-8d06-436b-9f76-13c3fd1a8011
md"""
**Defining Models**
"""

# ╔═╡ 7b56512c-31cd-4a15-9785-e6d37d8f36ab
begin
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
		(@load AdaBoostClassifier pkg = ScikitLearn)(),
		(@load BaggingClassifier pkg = ScikitLearn)(),
		(@load BayesianQDA pkg = ScikitLearn)(),
		(@load DecisionTreeClassifier pkg = DecisionTree)(),
		(@load ExtraTreesClassifier pkg = ScikitLearn)(),
		(@load KNNClassifier pkg = NearestNeighborModels)(),
		(@load RandomForestClassifier pkg = DecisionTree)(),
		(@load XGBoostClassifier pkg = XGBoost)()
	]

	# train kernel perceptron by itself later:
	#(@load KernelPerceptronClassifier pkg = BetaML)()
end

# ╔═╡ c0992d62-35a4-40e9-b3db-00a633a1c8c6
md"""
**Getting Cross-Validated Accuracy for Each Model**
"""

# ╔═╡ 599c496e-e226-4b69-8c2c-3428fbecab17
baseline_results = @pipe DataFrame(
	model_name = model_names,
	cv_acc = [MLJ.evaluate(clf, X, y, resampling=CV(shuffle=true, nfolds=10),
					   measure=accuracy, verbosity=0,
					   operation=predict_mode).measurement[1]
			  for clf ∈ all_models]
) |> sort(_, :cv_acc, rev = true)

# ╔═╡ e966ebb2-8e43-42c9-b0cd-8617f013b08a
md"""
**Plotting Baseline Results**
"""

# ╔═╡ 90473f1d-bc8f-4636-b1c6-c5c22828db24
@df baseline_results bar(
	:model_name,
	:cv_acc,
	orientation = :h,
	yflip = true,
	legend = false,
	xlabel = "Cross-Validated Accuracy",
	title = "Comparing Baseline Models"
)

# ╔═╡ a5f4a9b6-6b30-4d73-95ce-e2523876bff7
md"""
**Models I will tune:**
1. KNN
2. XGB
3. AdaBoost
"""

# ╔═╡ e295e34c-66ba-40b6-8d06-698f7a11b823
md"""
### Hyperparameter Tuning
"""

# ╔═╡ 4fb06a62-bf68-46b9-99d9-04bfee2a9f67
md"""
#### KNN
"""

# ╔═╡ 5b3e08f8-7888-4187-bd9b-bfdc28d97a95


# ╔═╡ fee587cd-5351-46f9-82ae-711b9e5333b4
begin
	knn_mod = (@load KNNClassifier pkg = NearestNeighborModels)()

	nn_range = range(knn_mod, :K, lower = 2, upper = 50)

	tuned_knn = TunedModel(model=knn_mod,
						   tuning=Grid(resolution=49),
						   resampling=CV(shuffle=true, nfolds=10),
						   ranges=[nn_range],
						   operation=predict_mode,
						   measure=accuracy)

	tuned_knn_mach = machine(tuned_knn, X, y)

	MLJ.fit!(tuned_knn_mach)

	knn_params = tuned_knn_mach |> fitted_params
	knn_report = tuned_knn_mach |> report

	knn_params.best_model
	knn_report.best_history_entry #63.62 accuracy
end

# ╔═╡ 0a0da777-32c6-49fa-aac5-afb82a445bc5


# ╔═╡ aa9f3767-2f1b-47d1-a624-2a522e65eb2a
begin
	
	accs = [knn_report.history[i].measurement[1] for i in 1:size(knn_report.history, 1)]
	
	plot(
		2:(size(accs, 1) + 1), 
		accs, 
		legend = false, 
		xlabel = "K",
		ylabel = "10-Fold CV Accuracy",
		title = "Tuning K by Cross-Validated Accuracy"
	)
end

# ╔═╡ 560ae576-cbde-41e3-83d7-f44e90540cb4
md"""
#### AdaBoost
"""

# ╔═╡ d26b6aab-cf01-48a9-93cf-a33868214896
begin
	ada = (@load AdaBoostClassifier pkg = ScikitLearn)()
	
	n_estimators_range = range(ada, :n_estimators, lower = 25, upper = 200)
	learning_rate_range = range(ada, :learning_rate, lower = 0.5, upper = 2)

	tuned_ada = TunedModel(model=ada,
						   tuning=Grid(resolution=11),
						   resampling=CV(shuffle=true, nfolds=10),
						   ranges=[n_estimators_range, learning_rate_range],
						   operation=predict_mode,
						   measure=accuracy)

	tuned_ada_mach = machine(tuned_ada, X, y)

	MLJ.fit!(tuned_ada_mach)

	ada_params = tuned_ada_mach |> fitted_params
	ada_report = tuned_ada_mach |> report

	ada_params.best_model
	ada_report.best_history_entry #63.62 accuracy
end

# ╔═╡ fd9c76cd-ae5b-4d5c-95a2-5252198620ee
md"""
#### XGBoost
"""

# ╔═╡ 949d95e1-e73b-4f48-b038-8ee43a45a602
begin
	xgb = (@load XGBoostClassifier pkg = XGBoost)()
end

# ╔═╡ Cell order:
# ╟─3b1b2020-d32b-11eb-1512-b74bcf73efee
# ╠═09a2ce5d-55b8-489c-8f67-5dccdb2d75c9
# ╟─7b8df1d5-ac46-4fe0-9f8d-7f6059d29c6c
# ╟─076bc9f3-8dfe-4307-b169-ac9a558dc411
# ╠═3c781196-8ad2-4de3-8398-05406b1b3c35
# ╟─67433b97-713e-430c-9be3-4cd5d46dfc21
# ╠═a81491b1-c799-4381-93fa-064abcdf6315
# ╟─5adca38e-02d2-4967-812d-fcf66b3e563a
# ╟─c7e0cd43-68a2-4458-a504-deaee4004176
# ╠═31afd76c-0610-4131-8123-d56eea703925
# ╟─0326b7da-1955-4de0-a94b-f59940c8056b
# ╠═d24dac74-e9fc-42d4-a781-8de32f3bf44a
# ╠═376c0560-8dee-482c-a4db-4748244f27c0
# ╟─eef1671e-3fd0-4464-854a-c4c342545ad7
# ╟─8e0a7fd6-43fe-4c3a-a089-6ca949c94f37
# ╠═f80df2ec-6cf2-430e-afea-0698935d3c1d
# ╟─2cb1c2ac-1e0d-472a-8550-4a5b0dfb08d9
# ╠═a7683b00-4f50-4290-a62e-27d181ff70ad
# ╟─d347c739-a68e-4881-9a6b-b6c7de8c711c
# ╟─9a88c3e5-75b6-4827-9287-8e7bdd76eaa1
# ╠═2f143245-529d-48df-8ad0-7c42814128b7
# ╟─99fb27dc-18cb-419b-b89e-7d8e7710aa58
# ╠═782159ec-e6c6-40e8-9007-2ff4c29d159d
# ╟─a8c63877-d106-4cc5-97fd-ec2b16a06a4b
# ╠═b8662c8c-4c8a-4326-8246-879aa6dec207
# ╟─732e5fd0-c190-48d4-aa9e-82a4184af6cb
# ╠═5e55c735-4e10-4103-8919-361366ef8edb
# ╠═3ceb40dc-23d8-461a-9cec-fc5653719ee3
# ╠═f1145ffe-4e35-4d55-b620-aca82bb3a76d
# ╟─5b629a72-b4fd-41aa-be02-eaab904f810e
# ╠═c49cb7bd-80d3-456a-818b-380c52c29bb3
# ╟─ac82eb5c-898c-4bde-9fd8-92711b016d80
# ╠═b6401293-7a4c-46a2-b935-2855cb47f2f4
# ╟─0055bf18-af67-4105-98fe-dab8fbdc3e5d
# ╠═7a1101f1-d78c-4300-91e3-f017c451fa98
# ╟─df238f00-6e4e-4b3c-8075-6a846e9e959c
# ╠═521628e0-3614-44bf-a063-b54f5f373197
# ╟─f55876b0-9196-4b7e-8f35-8925d15190c6
# ╟─04bc5320-6fb2-44ee-a7b8-bcf7191cafa6
# ╠═f9664918-12cd-404d-93a4-9e3aa2e81ddf
# ╟─015b282b-8d06-436b-9f76-13c3fd1a8011
# ╠═7b56512c-31cd-4a15-9785-e6d37d8f36ab
# ╟─c0992d62-35a4-40e9-b3db-00a633a1c8c6
# ╠═599c496e-e226-4b69-8c2c-3428fbecab17
# ╟─e966ebb2-8e43-42c9-b0cd-8617f013b08a
# ╠═90473f1d-bc8f-4636-b1c6-c5c22828db24
# ╟─a5f4a9b6-6b30-4d73-95ce-e2523876bff7
# ╟─e295e34c-66ba-40b6-8d06-698f7a11b823
# ╟─4fb06a62-bf68-46b9-99d9-04bfee2a9f67
# ╠═5b3e08f8-7888-4187-bd9b-bfdc28d97a95
# ╠═fee587cd-5351-46f9-82ae-711b9e5333b4
# ╠═0a0da777-32c6-49fa-aac5-afb82a445bc5
# ╠═aa9f3767-2f1b-47d1-a624-2a522e65eb2a
# ╟─560ae576-cbde-41e3-83d7-f44e90540cb4
# ╠═d26b6aab-cf01-48a9-93cf-a33868214896
# ╠═fd9c76cd-ae5b-4d5c-95a2-5252198620ee
# ╠═949d95e1-e73b-4f48-b038-8ee43a45a602
