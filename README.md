# Model Registry with MLFlow

This example shows the [MLFlow Metrics Tracking](https://mlflow.org/docs/latest/tracking.html) and  [MLFlow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html#) workflows

The repository is based on the [mlflow_monitoring](https://gith ub.com/evidentlyai/evidently/tree/main/examples/integrations/mlflow_monitoring) integration example from [Evidently](https://www.evidentlyai.com/)

![Model Registry with MLFlow](static/banner.png)

--------
Repository Structure

    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- Configs directory
    ├── data               <- Datasets
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Example Jupyter Notebook
    └── static             <- Assets for docs 
     

## :woman_technologist: Installation

### 1. Fork / Clone this repository

Get the tutorial example code:

```bash
git clone git@gitlab.com:mlrepa/mr/mr-1-bike-sharing-mlflow
cd bike-sharing-mlflow
```


### 2. Create a virtual environment

- This example requires Python 3.9 or above 

```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


### 4 - Download data

This is a preparation step. Load data from [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) to the `data/` directory

```bash 
python src/load_data.py              
```

## :tv: Run MLflow UI

```bash
mlflow ui
``` 
And then navigate to [http://localhost:5000](http://localhost:5000) in your browser


## 🎓 Run the tutorial
```bash
jupyter lab
``` 


## Acknowledgments

The dataset used in the example is downloaded from: https://www.kaggle.com/c/bike-sharing-demand/data?select=train.csv
- Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
- More information about the dataset can be found in UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset