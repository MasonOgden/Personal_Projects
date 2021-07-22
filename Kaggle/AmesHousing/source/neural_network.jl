#%% Packages
import CSV, MLJFlux
using DataFrames, Plots, MLJ, CategoricalArrays, Flux
using Impute: srs
using StatsPlots: @df
using Chain: @chain

code_folder = "source"

include(joinpath(code_folder, "functions.jl"))

#%% Reading and Cleaning Data

dataset_dir = "datasets"

df = @chain dataset_dir joinpath("train.csv") CSV.File DataFrame clean_data
#test = @chain dataset_dir joinpath("tesadd t.csv") CSV.File DataFrame clean_data
X_data = select(df, Not([:SalePrice, :Id]))

y_data = float.(df[!, :SalePrice])

#train, test = @chain y_data begin
#    eachindex
#    partition(0.75, shuffle=true)
#end

#%% Defining Networks

hidden_layer_range = [1, 2, 3, 4, 5]
width_range = [25, 50, 100, 150, 200]

nn_grid = collect(Base.product(hidden_layer_range, width_range))

nn_pipeline_names = vec([
    "Depth: $n_hidden_layers, Width = $hidden_layer_width" for (n_hidden_layers, hidden_layer_width) ∈ nn_grid
])

nn_pipelines = [
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(1, 25)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(1, 50)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(1, 100)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(1, 150)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(1, 200)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(2, 25)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(2, 50)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(2, 100)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(2, 150)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(2, 200)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(3, 25)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(3, 50)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(3, 100)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(3, 150)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(3, 200)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(4, 25)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(4, 50)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(4, 100)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(4, 150)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(4, 200)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(5, 25)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(5, 50)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(5, 100)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(5, 150)
        )
    ),
    @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(5, 200)
        )
    )
]

#%% Evaluating Networks

nn_results = @chain DataFrame(
    model = nn_pipeline_names,
    val_rmse = [
        evaluate(
            nn,
            X_data,
            y_data,
            resampling = Holdout(fraction_train = 0.75, shuffle = true),
            verbosity = 0,
        ).measurement[1] for nn ∈ nn_pipelines
    ],
) begin
    sort(:val_rmse)
end

# Best combos (based on validation RMSE)
    # Depth = 4, Width = 25, Validation RMSE = $23,129.1
    # Depth = 1, Width = 100, Validation RMSE = $23,536.9
    # Depth = 5, Width = 150, Validation RMSE = $24,871.5

### Hyperparameter Tuning

#%% Model 1

𝘈 = @pipeline(
        # get data ready for modeling
        X -> ready_data(X, X_data),
        # Standardize numeric variables
        Standardizer(count = true),
        # Dummify categorical variables
        OneHotEncoder(ordered_factor = false, drop_last = true),
        # load into model
        (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
            builder = MultiLayerBuilder(4, 25)
        )
    )

epochs_range = range(Int64, :(neural_network_regressor.epochs), lower = 7, upper = 15)
batch_size_range = range(Int64, :(neural_network_regressor.batch_size), lower = 1, upper = 10)
λ_range = range(Float64, :(neural_network_regressor.lambda), lower = 0.0, upper = 0.2)

𝘈_tune = @chain 𝘈 begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 5),
        resampling = Holdout(fraction_train = 0.75, shuffle = true),
        ranges = [epochs_range, batch_size_range, λ_range]
    )
    machine(X_data, y_data)
    fit!
end

𝘈_params = fitted_params(𝘈_tune)
𝘈_report = report(𝘈_tune)

𝘈_params.best_model # epochs = 9, batch_size = 1, lambda = 0.2
𝘈_report.best_history_entry # Validation RMSE = $40,710.43

#%% Model 2

𝘉 = @pipeline(
    # get data ready for modeling
    X -> ready_data(X, X_data),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
        builder = MultiLayerBuilder(1, 100)
    )
)

𝘉_tune = @chain 𝘉 begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 5),
        resampling = Holdout(fraction_train = 0.75, shuffle = true),
        ranges = [epochs_range, batch_size_range, λ_range]
    )
    machine(X_data, y_data)
    fit!
end

𝘉_params = fitted_params(𝘉_tune)
𝘉_report = report(𝘉_tune)

𝘉_params.best_model # epochs = 13, batch_size = 1, lambda = 0.05
𝘉_report.best_history_entry # Validation RMSE = $27,691.89

#%% Model 3

𝘊 = @pipeline(
    # get data ready for modeling
    X -> ready_data(X, X_data),
    # Standardize numeric variables
    Standardizer(count = true),
    # Dummify categorical variables
    OneHotEncoder(ordered_factor = false, drop_last = true),
    # load into model
    (@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0)(
        builder = MultiLayerBuilder(5, 150)
    )
)

𝘊_tune = @chain 𝘊 begin
    TunedModel(
        model = _,
        tuning = Grid(resolution = 5),
        resampling = Holdout(fraction_train = 0.75, shuffle = true),
        ranges = [epochs_range, batch_size_range, λ_range]
    )
    machine(X_data, y_data)
    fit!
end

𝘊_params = fitted_params(𝘊_tune)
𝘊_report = report(𝘊_tune)

𝘊_params.best_model # epochs = 7, batch_size = 6, lambda = 0.05
𝘊_report.best_history_entry # $27,565.00

#%% Saving Best Model

model_dir = "models"

@chain 𝘈 begin
    machine(X_data, y_data)
    fit!
    MLJ.save(joinpath(model_dir, "nn_d4_w25.jlso"), _)
end