# 🚀 MLFlow Metrics Tracking Tutorial - MLOps Python Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> A practical tutorial demonstrating MLFlow for metrics tracking and experiment management using a bike sharing dataset.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Development Workflow](#-development-workflow)
- [Usage](#-usage)
- [MLFlow UI](#-mlflow-ui)
- [Tutorial](#-tutorial)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

**A hands-on tutorial for learning MLFlow metrics tracking and experiment management.**

This project demonstrates how to use MLFlow for tracking machine learning experiments using a bike sharing demand prediction dataset. It emphasizes:

- **📊 Experiment Tracking**: Learn to track metrics, parameters, and artifacts
- **🔄 Model Versioning**: Understand model registry and versioning concepts
- **📈 Metrics Visualization**: Compare experiments and visualize results
- **🛠️ Best Practices**: Apply MLOps principles in practice

## ✨ Features

- **MLFlow 3.1 Integration**: Latest MLFlow features for experiment tracking and model registry
- **Enhanced Model Registry**: Improved model versioning and lifecycle management
- **Real Dataset**: Uses bike sharing demand dataset for practical learning
- **Jupyter Notebooks**: Interactive tutorials and examples
- **Model Training**: Example ML models with proper logging
- **Metrics Tracking**: Comprehensive metrics and parameter logging
- **Artifact Management**: Model and data artifact storage with enhanced metadata

## 📂 Project Structure

```text
mlflow-1-metrics-tracking/
├── data/                   # Data files
├── dev/                    # Development files
│   └── images/            # Development images
├── docs/                   # Documentation and images
├── mlartifacts/           # MLFlow artifacts storage
├── mlruns/                # MLFlow runs metadata
├── models/                # Trained models storage
├── notebooks/             # Jupyter notebooks for tutorials
│   └── mlruns/           # Notebook-specific MLFlow runs
├── src/                   # Source code
├── README.md              # This file
├── requirements-dev.txt   # Development dependencies
└── requirements.txt       # Project dependencies
```

## 🛠️ Prerequisites

Ensure you have the following installed:

- **Python 3.9+**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- **Jupyter**: For running tutorial notebooks

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://gitlab.com/risomaschool/tutorials-raif/mlflow-1-metrics-tracking.git
cd mlflow-1-metrics-tracking
```

### 2. Set Up the Environment

```bash
# Create virtual environment
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Download the Dataset

```bash
# Load bike sharing dataset
python src/load_data.py
```

### 4. Verify Installation

```bash
# Check if MLFlow is properly installed
mlflow --version
```

## 💻 Development Workflow

This project focuses on learning MLFlow through practical examples. Use the Makefile for easy commands:

### Key Development Commands

```bash
# Quick start workflow
make setup                  # Create virtual environment
make install               # Install dependencies
make load-data             # Download dataset
make mlflow-ui             # Start MLFlow UI (port 5001)
make jupyter               # Start Jupyter Lab

# Alternative manual commands
python src/load_data.py     # Download and prepare dataset
mlflow server --host 0.0.0.0 --port 5001  # Start MLFlow UI
jupyter lab                 # Start Jupyter for tutorials
```

## 🎯 Usage

### Running ML Experiments

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

# Start MLFlow experiment
mlflow.set_experiment("bike_sharing_experiment")

with mlflow.start_run():
    # Your model training code
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", 100)

    # Log metrics
    mlflow.log_metric("rmse", rmse_score)

    # Log model with MLflow 3.1 enhanced features
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="BikeSharing_RandomForest"
    )
```

## 📺 MLFlow UI

Start the MLFlow tracking server:

```bash
mlflow server --host 0.0.0.0 --port 5001
```

Then navigate to [http://localhost:5001](http://localhost:5001) in your browser to:

- 📊 **View Experiments**: Compare different model runs with MLflow 3.1's enhanced UI
- 📈 **Analyze Metrics**: Visualize training metrics over time with improved charts
- 🔍 **Inspect Artifacts**: Download models and other artifacts with better metadata
- 📋 **Compare Runs**: Side-by-side comparison of experiments with advanced filtering
- 🏷️ **Model Registry**: Manage model versions with enhanced lifecycle stages

## 🎓 Tutorial

Launch Jupyter Lab to access the interactive tutorials:

```bash
jupyter lab
```

The notebooks will guide you through:

1. **Data Exploration**: Understanding the bike sharing dataset
2. **MLFlow 3.1 Setup**: Setting up tracking and logging with latest features
3. **Experiment Tracking**: Logging parameters, metrics, and artifacts with enhanced metadata
4. **Model Comparison**: Comparing different models and hyperparameters using improved UI
5. **Model Registry**: Managing model versions and stages with MLflow 3.1's enhanced lifecycle management
6. **Advanced Features**: Exploring MLflow 3.1's new capabilities and improvements

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/improvement`
3. **Make your changes**
4. **Test your changes**: Ensure notebooks run correctly
5. **Commit your changes**: `git commit -m 'Add improvement'`
6. **Push to branch**: `git push origin feature/improvement`
7. **Open a Pull Request**

### Development Guidelines

- Follow Python best practices and PEP 8
- Add type hints to functions
- Document new features in notebooks
- Test MLFlow logging functionality
- Update README if adding new features

## 🙏 Acknowledgments

- **Dataset**: The bike sharing dataset is from [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data)
- **Research**: Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
- **Data Source**: [UCI Machine Learning Repository - Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- **Integration Example**: Based on the [mlflow_monitoring](https://github.com/evidentlyai/evidently/tree/main/examples/integrations/mlflow_monitoring) integration example from [Evidently AI](https://www.evidentlyai.com/)

---

**Happy learning with MLFlow! 🎉**
