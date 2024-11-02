# Tutorial: MLFlow for Data Science

![Model Registry with MLFlow](docs/images/mlflow-banner-1.png)

## 👩‍💻 Installation

Структура репозитория

    .
    ├── README.md           # Инструкции по установике и запуску проекта
    ├── data                # Файлы данных, используемые в проекте.
    ├── docs                # Файлы для документации (images)
    ├── models              # Директория для ML моделей 
    ├── notebooks           # Jupyter-блокноты 
    ├── requirements.txt    
    ├── src                 # Код примеров и утилит
    └── tutorial.md         # Тьюториал: MLFlow for Data Science



### 1. Fork / Clone this repository

```bash
git clone https://gitlab.com/risomaschool/tutorials-raif/mlflow-1-metrics-tracking.git
cd mlflow-1-metrics-tracking
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes:

- This example requires Python 3.9 or above 
- Tested on Mac OS


### 4 - Download data

Load data from [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) to the `data/` directory

```bash 
python src/load_data.py              
```

## 📺 Run MLflow UI

```bash
mlflow server --host 0.0.0.0 --port 5001
``` 
And then navigate to [http://localhost:5001](http://localhost:5001) in your browser


## 🎓 Run the tutorial

```bash
jupyter lab
```


## Acknowledgments

The repository is based on the [mlflow_monitoring](https://github.com/evidentlyai/evidently/tree/main/examples/integrations/mlflow_monitoring) integration example from [Evidently](https://www.evidentlyai.com/)

The dataset used in the example is downloaded from: https://www.kaggle.com/c/bike-sharing-demand/data?select=train.csv
- Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
- More information about the dataset can be found in UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
