Okay, let's implement those improvements, integrating the latest MLflow capabilities into the tutorial. I'll focus on the relevant code blocks and add explanations for the new features.

---

<!-- # Tutorial: MLflow Fundamentals for Data Science -->

![Model Registry with MLflow](docs/images/mlflow-banner-1.png)

# Tutorial: MLflow Fundamentals for Data Science
## 👀 Overview

- **What is this?** "MLflow Fundamentals for Data Science" is more than just a set of instructions; it's a comprehensive tutorial that will immerse you in the world of MLflow. It's a practical guide, full of examples and tips, to help you easily integrate MLflow into your Data Science projects.

- **Who is this tutorial for?** If you are a Data Scientist or ML Engineer, this material is definitely for you. Whether you're a beginner or experienced, you'll find plenty of useful and interesting information here.

**🎯 What will you learn?**

- How to track important metrics of your models using MLflow.
- How to create, compare, and manage ML experiments.
- How to understand metrics and artifacts from model runs.
- How to effectively use the Model Registry for model management.

- **How is it structured?** You won't have to search long for the information you need. The tutorial contains exhaustive code examples and step-by-step instructions in Markdown format.

- **How much time is needed?** Just 30 minutes – and you will significantly expand your knowledge and skills in MLflow.

## 📖 Table of Contents
- [Tutorial: MLflow Fundamentals for Data Science](#tutorial-mlflow-fundamentals-for-data-science)
  - [👀 Overview](#-overview)
  - [📖 Table of Contents](#-table-of-contents)
  - [👩‍💻 1 - Installation](#-1---installation)
  - [🚀 2 - Tracking Metrics in MLflow](#-2---tracking-metrics-in-mlflow)
    - [Step 1 - Train the Model and Calculate Metrics](#step-1---train-the-model-and-calculate-metrics)
    - [Step 2 - Logging Parameters, Metrics, and Artifacts](#step-2---logging-parameters-metrics-and-artifacts)
    - [Step 3 - MLflow UI](#step-3---mlflow-ui)
    - [Step 4 - Using the Saved Model for Generating Predictions](#step-4---using-the-saved-model-for-generating-predictions)
  - [🧑🏻‍🔬 3 - Managing Experiments and Runs](#-3---managing-experiments-and-runs)
    - [Step 1 - Data Preparation](#step-1---data-preparation)
    - [Step 2 - Creating Experiments with MLflow Client](#step-2---creating-experiments-with-mlflow-client)
    - [Step 4 - Tracking Metrics for K-Fold Cross Validation](#step-4---tracking-metrics-for-k-fold-cross-validation)
    - [Step 5 - Grouping Runs (Nested Runs)](#step-5---grouping-runs-nested-runs)
    - [Step 6 - Logging Metrics with Steps or Timestamps](#step-6---logging-metrics-with-steps-or-timestamps)
  - [🍱 4 - Model Management with MLflow Model Registry](#-4---model-management-with-mlflow-model-registry)
    - [How to register a model?](#how-to-register-a-model)
    - [How to find a registered model?](#how-to-find-a-registered-model)
    - [Working with MLflow Model Registry via API](#working-with-mlflow-model-registry-via-api)
  - [🔗 Additional Materials](#-additional-materials)
  - [💡 Suggested Improvements for MLflow 3.1+](#-suggested-improvements-for-mlflow-31)

## 👩‍💻 1 - Installation

First, install the pre-prepared example by following the instructions in the original repository's README.

**1. Fork / Clone this repository**

```bash
git clone https://gitlab.com/risomaschool/tutorials-raif/mlflow-1-metrics-tracking.git
cd mlflow-1-metrics-tracking
```

**2. Create a virtual environment**

This example requires Python version 3.9 or higher.

```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**3. Download data**

Download the data from [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) into the `data/` directory.

```bash
python src/load_data.py
```

**4. Start the MLflow UI**

To launch the MLflow UI, run the following command in your terminal:

```bash
mlflow ui
```

Then, navigate to [http://localhost:5000](http://localhost:5000) in your browser.

The MLflow UI allows you to view a list of experiments, as well as individual runs with their parameters and metrics. Metrics can be visualized as graphs, simplifying the analysis of trends and changes in metric values.

In the following sections of the tutorial, you will create several experiments and runs to see how this works!

![MLflow UI - Get Started](docs/images/1-mlflow-ui.png){width=800}

## 🚀 2 - Tracking Metrics in MLflow

> 💡 We'll start our introduction to MLflow in the Jupyter Notebook `notebooks/1-get-started.ipynb`.
> Launch Jupyter Lab or Jupyter Notebook and open this file.

In this part, we will explore the capabilities of tracking metrics using MLflow. MLflow is a library for experiment tracking and metric logging, installed as a Python package and easily integrated into project code.

Features:

- Log metrics, parameters, and artifacts in MLflow.
- Log, save, and load models using a local MLflow Tracking Server.
- Use the MLflow API to retrieve artifacts and experiment results.
- Run inference with models loaded as standard Python functions (pyfunc).

### Step 1 - Train the Model and Calculate Metrics

> 💡 Run the cells in the "Train model and calculate metrics" section. We will not go into detail about the code in this section. We hope the code examples are straightforward enough.

![Step 1 - Train the Model and Calculate Metrics](docs/images/2-1-train-model.png){width=800}

The example describes a simplified process for a typical Data Science project. As output, we have several objects related to the model experiment:

- Training and testing datasets for the model
- The model itself
- Model hyperparameters
- Model quality metrics

For convenient work with ML experiments, a method for organizing experiments and tracking results is needed. MLflow is an excellent option for these tasks! Tracking metrics and experiments in MLflow allows systematizing the machine learning model development process, ensuring reproducibility, comparison, and analysis of experiments.

### Step 2 - Logging Parameters, Metrics, and Artifacts

> 💡 Make sure MLflow is running and accessible at [http://localhost:5000](http://localhost:5000).
> If not, return to the "Installation" section and launch the MLflow UI.

![Step 2 - Logging Parameters, Metrics, and Artifacts](docs/images/2-2-mlflow-setup.png){ width=600 }

**MLflow Setup**
To start working with MLflow, you need to set the URI of the running MLflow Tracking Server, through which metrics and parameters will be logged. In this example, we will use the local server available at `http://localhost:5000`.

```python
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```

**Logging parameters, metrics, and artifacts**:
To start tracking experiments, use `mlflow.start_run()`, which will create a new MLflow Run and log metrics and parameters into it. Within this block, various parameters and metrics can be logged:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error # Assuming these are available from training
from mlflow.models import infer_signature # Ensure this import is present

with mlflow.start_run() as run:
    # Log model parameters
    mlflow.log_param('model', 'RandomForest')
    mlflow.log_params({'random_state': 0, 'n_estimators': 50})

    # --- IMPROVEMENT 1: Utilize mlflow.evaluate() for Comprehensive Model Evaluation ---
    # Create evaluation dataset (X_test and y_test must be defined from previous steps)
    eval_data = pd.DataFrame(X_test)
    eval_data[target.name] = y_test # Assuming 'target' is a Series with a name attribute

    # Run MLflow evaluation, which automatically logs metrics and plots
    # 'me' (RMSE) and 'mae' are logged by default for 'regressor' type
    mlflow.evaluate(
        model=model, # Your trained RandomForest model
        data=eval_data,
        targets=target.name, # The name of your target column in eval_data
        model_type="regressor",
        evaluators=["default"], # Use default evaluators for regressor
        # You can add custom metrics or plots if needed:
        # extra_metrics=[MetricInfo(name="my_custom_metric", metric_fn=my_custom_metric_func, greater_is_better=True)],
        # custom_artifacts={"my_plot": my_plot_generator_func},
    )

    # --- Original (manual) metric logging replaced by mlflow.evaluate() ---
    # me = mean_squared_error(y_test, preds) # If you still need direct access for other uses
    # mae = mean_absolute_error(y_test, preds) # If you still need direct access for other uses
    # mlflow.log_metric('me', round(me, 3))
    # mlflow.log_metric('mae', round(mae, 3))

    # Log artifacts, including raw data
    mlflow.log_artifact("../data/raw_data.csv")
    # No need to log model.joblib separately if logging model via mlflow.sklearn.log_model()

    # --- IMPROVEMENT 4: Enhanced Logging of Plots and Figures ---
    # Assuming 'preds' is available from model.predict(X_test)
    preds = model.predict(X_test) # Ensure predictions are made for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, preds, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs. Predicted Values")
    mlflow.log_figure(fig, "actual_vs_predicted_plot.png")
    plt.close(fig) # Close the figure to free up memory

    # Set tags for easier experiment tracking
    mlflow.set_tag("random-forest", "Random Forest Classifier")

    # --- IMPROVEMENT 3: Explicit Input Example for infer_signature ---
    # Define model signature using X_train as the input example.
    # While infer_signature(X_train, model.predict(X_train)) works, providing a concrete input_example
    # to log_model can sometimes be more robust for schema inference.
    # Here, we infer the signature based on the training data and model's output on it.
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model itself
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="rf_model",
        signature=signature,
        input_example=X_train.head(5), # Use a small subset as input example
        registered_model_name="1-get-started-random-forest",
    )
```

This code block demonstrates using MLflow to log the results of an ML experiment. Let's break it down:

1.  **Start Experiment**:

    ```python
    with mlflow.start_run() as run:
        #...
    ```

    This line initiates a new experiment run in MLflow. The `with` statement ensures that everything within this block pertains to the current Run. After the block completes, the context is automatically closed.

2.  **Log Model Parameters**:

    ```python
    mlflow.log_param('model', 'RandomForest')
    mlflow.log_params({'random_state': 0, 'n_estimators': 50})

    ```

    Here, model parameters are logged. `log_param()` is used to record a single parameter, and `log_params()` for logging multiple parameters simultaneously.

3.  **Comprehensive Model Evaluation with `mlflow.evaluate()`**:
    -   This is a significant improvement. Instead of manually calculating `me` and `mae` and logging them individually, `mlflow.evaluate()` performs a more comprehensive evaluation.
    -   It automatically computes common regression metrics (like RMSE, MAE, R2) and logs them.
    -   Crucially, it also generates and logs standard plots (e.g., residual plots, prediction error plots) that are highly valuable for understanding model performance, without requiring you to write plotting code.
    -   The results of `mlflow.evaluate()` are logged within a *nested run* under the current run, making your evaluation process modular and easily viewable in the MLflow UI.

4.  **Log Artifacts**:

    ```python
    mlflow.log_artifact("../data/raw_data.csv")
    ```

    For logging files (artifacts) of the experiment, the path to the saved file is passed to the `log_artifact` method. For example, you can log the raw data (`raw_data.csv`).

    > ⚠️ Note that when logging a model using `log_artifact()`, you must pass the path to the file, not the model object itself! For logging models, it's more convenient to use `log_model()` (see below). Since we are using `mlflow.sklearn.log_model`, logging `model.joblib` separately is no longer necessary.

5.  **Enhanced Logging of Plots and Figures with `mlflow.log_figure()`**:
    -   This new line demonstrates how to log a Matplotlib figure directly. MLflow will convert the figure into an image file (e.g., PNG) and store it as an artifact, making it directly previewable in the MLflow UI.
    -   This is excellent for custom plots, data visualizations, or diagnostic plots that `mlflow.evaluate()` might not generate by default.

6.  **Set Tags**:

    ```python
    mlflow.set_tag("random-forest", "Random Forest Classifier")
    ```

    Setting tags helps categorize and easily identify experiments in MLflow.

7.  **Define Model Signature with `input_example`**:

    ```python
    signature = infer_signature(X_train, model.predict(X_train))
    ```

    A model signature defines the format of input and output data. This is useful for understanding what data the model expects as input and what it outputs. By providing `input_example=X_train.head(5)` to `log_model`, you explicitly tell MLflow a representative sample of inputs, which can be used for schema validation or serving example inference requests.

8.  **Log the Model Itself**:

    ```python
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="rf_model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name="1-get-started-random-forest",
    )
    ```

    This step saves the model to MLflow, including its signature and an explicit input example. This allows later easy loading and use of the model, and provides a clear representation of its expected inputs.

Overall, this code presents a more modern and comprehensive example of using MLflow for tracking all key results of an ML experiment, leveraging newer capabilities for evaluation and artifact logging.

### Step 3 - MLflow UI

After running the code in the previous section, you can navigate to the MLflow UI and see the experiment results!

MLflow saves all logged data to a unique directory for each run and stores them in a database. The MLflow web interface allows you to view a list of experiments, as well as individual runs with their parameters and metrics. Metrics can be visualized as graphs, simplifying the analysis of trends and changes in metric values.

By default, MLflow registers all metrics and artifacts in the **`Default`** experiment.

![Untitled](docs/images/2-3-mlflow-new-run.png){width=800}

Go to the newly created run and check that you successfully logged metrics (now likely under a nested `evaluate` run), and the `actual_vs_predicted_plot.png` in the artifacts section.

![Untitled](docs/images/2-3-mlflow-run-page-1.png){width=800}

### Step 4 - Using the Saved Model for Generating Predictions

> 💡 This part of the example may not be suitable for use in `production` processes. However, it can be useful for quick model testing and analysis.

![Untitled](docs/images/2-4-mlflow-download-model.png){width=800}

After the model has been trained and logged in MLflow, the next step is to load it and use it to obtain predictions. In this section, we will explore how to load a saved model as a universal Python function and use it for prediction on new data.

**Loading the model as a Python function**:

Instead of loading the model back in its native scikit-learn format with `mlflow.sklearn.load_model()`, we load the model as a universal Python function. This approach can be used for both online model serving and batch scoring.

```python
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
```

Here, `model_info.model_uri` contains the URI of the model saved in MLflow.

**Obtaining predictions for the test dataset**:

Now that the model is loaded, it can be used to obtain predictions on the `X_test` dataset.

```python
# Get predictions
predictions = loaded_model.predict(X_test)

# Save results to Pandas DataFrame
result = pd.DataFrame(X_test)
result["actual_class"] = y_test
result["predicted_class"] = predictions

# --- IMPROVEMENT 2: More Structured Logging for DataFrames/Tables with mlflow.log_table() ---
# Instead of saving to CSV and then logging as a generic artifact,
# use mlflow.log_table for structured data.
mlflow.log_table(data=result, artifact_file="predictions.json")
# You can choose other formats like "predictions.parquet" or "predictions.csv"
# The 'data' can be a Pandas DataFrame or a PyArrow Table.
```

![Untitled](docs/images/2-4-mlflow-predictions.png){width=800}

Loading and using a model for predictions is a critically important stage in the machine learning workflow. Using MLflow, you can effectively manage models and their deployment, now with enhanced structured logging of your prediction results.

## 🔬 3 - Managing Experiments and Runs

> 💡 In this section, you will work with the Jupyter Notebook `notebooks/2-manage-runs.ipynb`.

In the previous section, you logged experiment results using only `mlflow.start_run()`. In this section, you will learn to use another mechanism – the [MLflow Client](https://mlflow.org/docs/latest/getting-started/logging-first-model/step2-mlflow-client.html).

The `mlflow.client` module provides a Python interface for CRUD operations (creating, reading, updating, and deleting) with MLflow Experiments, Runs, Model Versions, and Registered Models. With it, you can:

- Initiate a new Experiment.
- Create new Experiment Runs.
- Log parameters, metrics, and tags for Runs.
- Register artifacts associated with runs, such as models, tables, plots, and more.

### Step 1 - Data Preparation

We use the same dataset as in the previous example. Run the cells in section `2. Prepare Data`.

### Step 2 - Creating Experiments with MLflow Client

Configure `MLFLOW_TRACKING_URI` and create an MLflow Client. The created `client` object can be used to create new experiments and runs.

```python
# Set up MLflow Client
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()
print(f"Client tracking uri: {client.tracking_uri}")
```

Experiments in MLflow group individual project runs. To create a new experiment `1-Train-K-Fold`, you can:

- Use `mlflow.set_experiment('1-Train-K-Fold')` (the simplest option).
- Use the MLflow Client (to get access to the experiment ID and other metadata).

Let's look at an example using the MLflow Client.

```python
# Get experiment by name, if it exists
experiment_id = client.get_experiment_by_name('1-Train-K-Fold')

# Create a new experiment if it doesn't exist
if not experiment_id:
    experiment_id = client.create_experiment('1-Train-K-Fold')

# Get experiment metadata information
experiment = client.get_experiment(experiment_id)
print(f"Name: {experiment.name}")
print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
```

As a result, a new experiment `1-Train-K-Fold` will appear in the MLflow UI.
![Untitled](docs/images/3-3-new-experiment.png){width=800}

### Step 4 - Tracking Metrics for K-Fold Cross Validation

> 💡 Go to the `Metrics Tracking for K-Fold Experiments` section in Jupyter Notebook `notebooks/2-manage-runs.ipynb`.

Now let's implement a model training scenario with K-Fold cross-validation and log metrics to the `1-Train-K-Fold` experiment. For each iteration (Fold), you need to:

- Define the training and testing datasets.
- Train the model using the training dataset.
- Make predictions using the trained model.
- Calculate model quality metrics.
- Log the trained model, metrics, and artifacts to MLflow.

Given that our dataset has a temporal dimension, we will follow the recommendations for [Time Series Split](https://scikit-learn.org/stable/modules/cross_validation.html).

![Untitled](docs/images/3-4-tseries-split.png){width=600}

Let's break down the example code from section **4. Metrics Tracking for K-Fold Experiments**:

```python
# Assuming necessary imports like pandas, sklearn.ensemble,
# sklearn.metrics, etc., are available.
import pandas as pd
from sklearn import ensemble
import matplotlib.pyplot as plt # For plotting

# Set experiment name
mlflow.set_experiment('1-Train-K-Fold')
# ... (definitions of raw_data, numerical_features, categorical_features, target, experiment_batches)

for k, date in enumerate(experiment_batches):

    # Define train data
    X_train = raw_data.loc[start_date_0:ref_end_data, numerical_features + categorical_features]
    y_train = raw_data.loc[start_date_0:ref_end_data, target]

    # Define test data
    current = raw_data.loc[date[0]:date[1]]
    X_test = current.loc[:, numerical_features + categorical_features]
    y_test = current[target]

    # Train model
    regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(X_train, y_train)

    # --- IMPROVEMENT 1: Utilize mlflow.evaluate() ---
    # Prepare data for evaluation
    eval_data = pd.DataFrame(X_test)
    eval_data[target.name] = y_test # Assuming 'target' is a Series with a name attribute

    # Start a new MLflow Run for each fold
    with mlflow.start_run() as run:

        # Log parameters specific to this fold
        mlflow.log_param("begin", date[0])
        mlflow.log_param("end", date[1])

        # Run MLflow evaluation for this fold
        # This will create a nested run and log metrics (me, mae, etc.) and plots
        mlflow.evaluate(
            model=regressor,
            data=eval_data,
            targets=target.name,
            model_type="regressor",
            evaluators=["default"],
            # Artifact path within this run. Default is "evaluators".
            # You can customize it like: artifact_path=f"fold_{k}_evaluation"
        )

        # --- IMPROVEMENT 4: Enhanced Logging of Plots and Figures for each fold ---
        preds_fold = regressor.predict(X_test) # Predictions for current fold
        fig_fold, ax_fold = plt.subplots(figsize=(7, 5))
        ax_fold.scatter(y_test, preds_fold, alpha=0.6)
        ax_fold.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax_fold.set_xlabel("Actual")
        ax_fold.set_ylabel("Predicted")
        ax_fold.set_title(f"Fold {k} Actual vs. Predicted")
        mlflow.log_figure(fig_fold, f"fold_{k}_actual_vs_predicted.png")
        plt.close(fig_fold) # Close the figure to free up memory

        # --- IMPROVEMENT 3: Explicit Input Example for infer_signature ---
        # Define model signature explicitly for this fold's model
        signature_fold = infer_signature(X_train, regressor.predict(X_train))

        # Log the trained model for this fold
        mlflow.sklearn.log_model(
            sk_model=regressor,
            artifact_path=f"fold_{k}_model",
            signature=signature_fold,
            input_example=X_train.head(3), # Smaller input example for each fold's model
            # registered_model_name is typically not used for individual fold models
        )

    # Update reference end date
    ref_end_data = date[1]
```

The difference between this example and the previous one is that here we have a loop for training the model across folds. For each fold, `mlflow.evaluate()` is now used to calculate and log metrics and plots.

```python
for k, date in enumerate(experiment_batches):
    ...

    # Start a new MLflow Run
    with mlflow.start_run() as run:
        ...
        mlflow.evaluate(...) # New comprehensive evaluation
        mlflow.log_figure(...) # Log custom plot for each fold
        mlflow.sklearn.log_model(...) # Log model for this fold
        ...
```

It is important to note that a new run is created in MLflow for each fold. Within each run, `mlflow.evaluate()` creates its own nested run to log detailed evaluation results and plots. As a result, you will get a list of runs within the "1-Train-K-Fold" experiment, each with a nested evaluation run and its own logged model and plot.

![Untitled](docs/images/3-5-kfold-runs.png){width=800}

You can configure which columns to display on the dashboard. Simply go to the "Columns" section and select the necessary metrics and parameters.

![Untitled](docs/images/3-5-set-up-columns.png){width=800}

### Step 5 - Grouping Runs (Nested Runs)

> 💡 Go to the `Nested Runs` section in Jupyter Notebook `notebooks/2-manage-runs.ipynb`.

In the previous example, each experiment run created multiple entries (Runs) in the dashboard, which might not be very convenient. After all, you will likely be more interested in the metrics on the last fold. It is by the metrics on the last fold that you will compare experiment runs.

In this step, you will make some improvements:

- Configure run grouping (Nested Runs)
- Add the batch end date for naming folds
- Configure logging metric and parameters for each fold
- Configure logging the model only for the last fold

To group runs in MLflow, use the "nested" runs feature. The code snippet below demonstrates the script structure for implementing this.

- First, initiate the main Run (let's call it Parent). The Parent Run is a single experiment run encompassing all folds.
- Inside the Parent Run, initiate several nested (or Nested) runs for each fold. To do this, pass the argument `nested=True` to the `mlflow.start_run()` function.
- Pass the argument `run_name` to the `mlflow.start_run()` function, to set a name for the Nested Run.

```python
# Start a new Run (Parent Run)
with mlflow.start_run() as run:

    # Train model for each fold
    for k, date in enumerate(experiment_batches):

        # Start a Child Run for each fold
        with mlflow.start_run(run_name=date[1], nested=True) as child_run:

            # Log to Child Run (Fold)
            ...

    # Log to Parent Run
    ...
```

Let's discuss the implementation details. Inside the Parent Run, you start training the model with K-Fold validation. For each fold, a separate nested child run is created, into which you log metrics.

At the same time, metrics for each fold are added to the object `metrics`, so that they can be aggregated subsequently.

```python
# Update metrics with metrics for each Fold
metrics = {}

# Train model for each batch (K-Fold)
for k, date in enumerate(experiment_batches):

    # Model training code, performance reporting, and metric extraction...

    # Save metrics for this batch to a dictionary
    metrics.update({date[1]: {'me': me, 'mae': mae}})
```

Finally, after cross-validation is complete, for the Parent Run:

- Save the model and log it as a Parent Run artifact.
- Calculate aggregated metrics across all folds and log them to the Parent Run.

```python
# Assuming necessary imports like pandas, sklearn.ensemble,
# sklearn.metrics, joblib, matplotlib.pyplot are available.

# Start a new Run (Parent Run)
with mlflow.start_run() as run:

    # Store metrics of each fold in one object for later aggregation
    all_fold_metrics = []
    final_regressor = None # To store the model from the last fold

    # Train model for each batch (K-Fold)
    for k, date in enumerate(experiment_batches):
        # Define train and test data for the current fold
        X_train_fold = raw_data.loc[start_date_0:ref_end_data, numerical_features + categorical_features]
        y_train_fold = raw_data.loc[start_date_0:ref_end_data, target]

        current_test = raw_data.loc[date[0]:date[1]]
        X_test_fold = current_test.loc[:, numerical_features + categorical_features]
        y_test_fold = current_test[target]

        # Train model for the current fold
        regressor_fold = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
        regressor_fold.fit(X_train_fold, y_train_fold)
        preds_fold = regressor_fold.predict(X_test_fold)

        # Calculate metrics for the current fold
        me_fold = mean_squared_error(y_test_fold, preds_fold)
        mae_fold = mean_absolute_error(y_test_fold, preds_fold)

        all_fold_metrics.append({'me': me_fold, 'mae': mae_fold})
        final_regressor = regressor_fold # Keep track of the last trained model

        # Start a Child Run for each Fold
        with mlflow.start_run(run_name=f"Fold_{date[1]}", nested=True) as child_run: # Renamed run_name for clarity
            # Log parameters for the child run
            mlflow.log_param("begin", date[0])
            mlflow.log_param("end", date[1])

            # --- IMPROVEMENT 1: Utilize mlflow.evaluate() in Nested Runs ---
            # Prepare data for evaluation
            eval_data_fold = pd.DataFrame(X_test_fold)
            eval_data_fold[target.name] = y_test_fold

            # Run MLflow evaluation for this fold
            mlflow.evaluate(
                model=regressor_fold,
                data=eval_data_fold,
                targets=target.name,
                model_type="regressor",
                evaluators=["default"],
                # A custom artifact path for this evaluation within the child run:
                artifact_path="evaluation_results"
            )

            # --- IMPROVEMENT 4: Enhanced Logging of Plots and Figures in Nested Runs ---
            fig_fold, ax_fold = plt.subplots(figsize=(7, 5))
            ax_fold.scatter(y_test_fold, preds_fold, alpha=0.6)
            ax_fold.plot([y_test_fold.min(), y_test_fold.max()], [y_test_fold.min(), y_test_fold.max()], 'k--', lw=2)
            ax_fold.set_xlabel("Actual")
            ax_fold.set_ylabel("Predicted")
            ax_fold.set_title(f"Fold {k+1} Actual vs. Predicted") # Use k+1 for 1-based indexing
            mlflow.log_figure(fig_fold, f"fold_{k+1}_actual_vs_predicted.png")
            plt.close(fig_fold)

    # After all folds are complete, log to the Parent Run

    # --- IMPROVEMENT 3: Explicit Input Example for infer_signature for the final model ---
    # Define signature for the final model (from the last fold)
    signature_final = infer_signature(X_train_fold, final_regressor.predict(X_train_fold)) # Using X_train_fold from last iteration

    # Log the final model (from the last fold) as an artifact of the Parent Run
    mlflow.sklearn.log_model(
        sk_model=final_regressor,
        artifact_path="final_model", # Path within the parent run
        signature=signature_final,
        input_example=X_train_fold.head(5), # Small input example
        registered_model_name="2-kfold-random-forest", # Register the final model
    )

    # Log averaged metrics in the Parent Run
    average_run_metrics = pd.DataFrame(all_fold_metrics).mean().round(3).to_dict()
    mlflow.log_metrics(average_run_metrics)

    # --- IMPROVEMENT 2: More Structured Logging for aggregated evaluation results ---
    # Example: You might aggregate the evaluation metrics from all folds into a DataFrame
    # and log it here. For simplicity, we'll log the average metrics DataFrame.
    avg_metrics_df = pd.DataFrame([average_run_metrics])
    mlflow.log_table(data=avg_metrics_df, artifact_file="average_kfold_metrics.json")
```

Let's see how the results are displayed in MLflow.

![Untitled](docs/images/3-5-ui.png){width=800}

> ⚠️ Please note that:
>
- In the parent Run, averaged metrics are displayed (aggregated across all folds).
- The trained model from the *last fold* is saved in the parent Run and registered in the Model Registry.
- For each Nested Run (fold), `mlflow.evaluate()` creates its own sub-run for detailed evaluation results, including metrics and plots.
- A custom plot `fold_X_actual_vs_predicted.png` is logged for each nested run.
- `average_kfold_metrics.json` is logged as a structured table in the parent run.

Thus, you can use MLflow to organize model training with cross-validation and logically separate metadata between parent and nested runs. This gives you the ability to delve into the details of each fold while maintaining an overall overview of the experiment, now with more comprehensive and structured logging.

This approach can also be used when training other models with a large number of iterations and epochs.

### Step 6 - Logging Metrics with Steps or Timestamps

> 💡 Go to the `Log metrics by steps or timestamps` section in Jupyter Notebook `notebooks/2-manage-runs.ipynb`.

Logging metrics by steps or temporal timestamps in MLflow allows for more precise tracking of changes in model performance over training time.

Unlike the previous step, this approach does not create separate Nested Runs for each fold, but logs metrics within a single run.

```python
# Assuming necessary imports are available
import datetime
import time

# Set up MLflow Client
# ... (mlflow_tracking_uri and client definition)

# Set experiment name
mlflow.set_experiment('3-Metrics-by-steps')

# Set experiment variables
model_path = Path('../models/model.joblib')
# ... (raw_data, numerical_features, categorical_features, target, experiment_batches, start_date_0, end_date_0 definitions)

# Start a new MLflow Run
with mlflow.start_run() as run:

    # Store the latest regressor model for logging at the end
    latest_regressor = None

    # Run model train for each batch (K-Fold)
    for k, date in enumerate(experiment_batches):

        # Define train and test data for the current fold
        X_train_fold = raw_data.loc[start_date_0:date[1], numerical_features + categorical_features] # Adjusting train data end date
        y_train_fold = raw_data.loc[start_date_0:date[1], target]

        current_test = raw_data.loc[date[0]:date[1]]
        X_test_fold = current_test.loc[:, numerical_features + categorical_features]
        y_test_fold = current_test[target]

        # Train model
        regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
        regressor.fit(X_train_fold, y_train_fold)
        preds = regressor.predict(X_test_fold)
        latest_regressor = regressor # Keep track of the latest trained model

        # Calculate timestamp for metric logging
        timestamp = time.mktime(datetime.datetime.strptime(date[1], "%Y-%m-%d %H:%M:%S").timetuple())

        # --- IMPROVEMENT 1: Utilize mlflow.evaluate() in this single run ---
        # Prepare data for evaluation
        eval_data_fold = pd.DataFrame(X_test_fold)
        eval_data_fold[target.name] = y_test_fold

        # Run MLflow evaluation for this step/timestamp
        # This will create a NESTED run under the current run, logging metrics and plots.
        # This structure helps keep the main run clean while providing detailed evaluation per step.
        mlflow.evaluate(
            model=regressor,
            data=eval_data_fold,
            targets=target.name,
            model_type="regressor",
            evaluators=["default"],
            # Specify the step/timestamp for the evaluation results
            step=k,
            # timestamp=int(timestamp)*1000, # MLflow evaluate uses its own timestamping
            artifact_path=f"evaluation_step_{k}" # Custom path for each step's eval results
        )

        # Manual logging of specific metrics if needed, using step or timestamp
        # client.log_metric(run.info.run_id, 'me', round(mean_squared_error(y_test_fold, preds), 3), timestamp=int(timestamp)*1000)
        # client.log_metric(run.info.run_id, 'mae', round(mean_absolute_error(y_test_fold, preds), 3), step=k)

        # --- IMPROVEMENT 4: Enhanced Logging of Plots and Figures for each step ---
        fig_step, ax_step = plt.subplots(figsize=(7, 5))
        ax_step.scatter(y_test_fold, preds, alpha=0.6)
        ax_step.plot([y_test_fold.min(), y_test_fold.max()], [y_test_fold.min(), y_test_fold.max()], 'k--', lw=2)
        ax_step.set_xlabel("Actual")
        ax_step.set_ylabel("Predicted")
        ax_step.set_title(f"Step {k+1} Actual vs. Predicted")
        mlflow.log_figure(fig_step, f"step_{k+1}_actual_vs_predicted.png", step=k) # Log figure with step
        plt.close(fig_step)

    # Log the final model from the last step to the main run
    # --- IMPROVEMENT 3: Explicit Input Example for infer_signature for the final model ---
    if latest_regressor:
        signature_final = infer_signature(X_train_fold, latest_regressor.predict(X_train_fold)) # Using X_train_fold from last iteration
        mlflow.sklearn.log_model(
            sk_model=latest_regressor,
            artifact_path="final_model_by_steps",
            signature=signature_final,
            input_example=X_train_fold.head(5),
            registered_model_name="3-metrics-by-steps-final-model",
        )

    # Log parameters (these are usually run-level)
    mlflow.log_param("begin_data_range", experiment_batches[0][0])
    mlflow.log_param("end_data_range", experiment_batches[-1][1])
```

When logging metrics, an additional `step` or `timestamp` parameter is specified, which allows tracking metric changes depending on time or training iteration. In this improved version, `mlflow.evaluate()` handles much of this automatically by creating nested runs for each step, and `mlflow.log_figure()` also supports logging with `step`. As a result, MLflow saves the metrics and plots in a detailed, time-series-like fashion, which can be visualized in the UI.

![Untitled](docs/images/3-6-metrics-by-steps.png){width=800}

Logging metrics by steps or temporal timestamps is a key component for detailed analysis and tracking of model performance during training. This is especially important in cases of working with time-series data or when using complex validation methods, such as K-Fold Cross-Validation, now enhanced with MLflow's modern evaluation capabilities.

## 🍱 4 - Model Management with MLflow Model Registry

> 💡 In this section, we will work with Jupyter Notebook `notebooks/3-model-registry.ipynb`.

The MLflow Model Registry (MLflow Model Registry) is a centralized model store, a set of APIs, and a UI for collaborative management of the full MLflow model lifecycle. It provides model versioning, allows assigning aliases, adding tags and annotations, and tracking model lineage (from which MLflow experiment and run the model was created).

Key features of MLflow Model Registry:

1.  A centralized repository for storing all models of your project.
2.  Model version management.
3.  Adding tags and annotations, which can be useful for documenting and classifying models.

### How to register a model?

Follow the steps below to register your MLflow model in the Model Registry.

**Open the MLflow Run page**
Navigate to the details page of the MLflow Run that contains the logged MLflow model you wish to register. Select the model folder containing the desired MLflow model in the "Artifacts" section.

![Untitled](docs/images/4-1-model-registry-1.png){width=800}

Click the "Register Model" button, and a form will appear.

In the "Model" dropdown menu on the form, you can either select "Create New Model," which will create a new registered model with your MLflow model as its initial version, or select an existing registered model, under which your model will be registered as a new version. The screenshot below shows the process of registering an MLflow model in a new registered model named "iris_model_testing."

![Untitled](docs/images/4-1-model-registry-2.png){width=800}

### How to find a registered model?

After registering your models in the Model Registry, you can find them in the following ways.

1.  Go to the Registered Models page.
    This page contains links to your registered models and corresponding model versions.

    ![Untitled](docs/images/4-1-model-registry-3.png){width=800}

2.  Go to the Artifacts Section on your MLflow Run page.
    Click on the model folder, then click on the model version in the top right corner to view the version created from this model.

    ![Untitled](docs/images/4-1-model-registry-4.png){width=800}

### Working with MLflow Model Registry via API

An alternative way to interact with the Model Registry is by using the MLflow `model flavor` interface or the MLflow client tracking API. Specifically, you can register a model during an MLflow experiment run or after all your experimental runs.

**How to register a model via API?**
For automatic model registration, specify the model name in the registry as the `registered_model_name` parameter of the `log_model()` method.

```python
mlflow.sklearn.log_model(
    sk_model=model_rf,
    registered_model_name="RandomForest",
)
```

- If no model with such a name exists in the registry, the method will register a new model and assign `Version 1`.
- If a registered model with that name already exists, the method creates a new model version.

Run the following code cell **3 times**, to register 3 versions of the "RandomForest" model.

```python
from mlflow.models import infer_signature
# Assume y_test and preds_rf (predictions from your RandomForest model) are available
# from a previous step or generated here for demonstration.
# For example:
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# model_rf = RandomForestRegressor(random_state=0, n_estimators=50)
# model_rf.fit(X_train, y_train)
# preds_rf = model_rf.predict(X_test)


with mlflow.start_run() as run:

    # --- IMPROVEMENT 3: Explicit Input Example for infer_signature ---
    # Define model signature using actual input and output examples from your test data.
    signature = infer_signature(X_test, preds_rf)

    # Log the sklearn model and register as a new version
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="RandomForest",
        signature=signature,
        input_example=X_test.head(5), # Explicitly provide a small input example
        registered_model_name="RandomForest",
    )
```

**How to find registered models?**
Use `client.search_registered_models()` to search and display registered models.

```python
from pprint import pprint

# List and search MLflow models
for rm in client.search_registered_models():
    pprint(dict(rm), indent=4)
```

**How to change the model stage to Production?**
A model goes through different lifecycle stages: from development to testing and production. To specify the desired stage, use the `transition_model_version_stage()` method.

```python
client.transition_model_version_stage(
    name="RandomForest",
    version=3,
    stage="Production"
)
```

**How to load and use a model from the Model Registry?**
To load a model, you will need the model URI and its `flavor`. The model URI consists of the model name and version. The model flavor is the framework-specific format in which the model is saved in MLflow. For example, for `scikit-learn` models, it's the pickle format.

```python
model_uri = "models:/RandomForest/3"
loaded_model = mlflow.sklearn.load_model(model_uri)
loaded_model
```

**How to unregister, delete, and archive a model?**
You can unregister, delete, and archive a model using the `delete_registered_model()`, `delete_model_version()`, and `transition_model_version_stage()` methods.

For example, to delete version 1 of the RandomForest model, use:

```python
# Delete version 1 of the model
client.delete_model_version(
    name="RandomForest", version=1,
)
```

To archive version 2 of the RandomForest model, use:

```python
client = MlflowClient()
client.transition_model_version_stage(
    name="RandomForest", version=2, stage="Archived"
)
```

In this section, you learned how to use the MLflow API to interact with the model registry, including registering, searching, changing stages, loading and using, as well as unregistering, deleting, and archiving models. Now you can automate a large part of model lifecycle management processes for MLOps.

## 🔗 Additional Materials

- Getting Started with MLflow: [https://mlflow.org/docs/latest/getting-started/index.html](https://mlflow.org/docs/latest/getting-started/index.html)
- MLflow Model Registry: [https://mlflow.org/docs/latest/model-registry.html#ui-workflow](https://mlflow.org/docs/latest/model-registry.html#ui-workflow)


[⬆️ Table of Contents](#-table-of-contents)
