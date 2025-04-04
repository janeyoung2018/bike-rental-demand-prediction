# Bike Renting Prediction

This machine learning project predicts bike rental demand using historical trends and environmental factors. It's built to support bike-sharing services in optimizing fleet allocation, improving bike availability, and uncovering usage patterns.

The project includes:

A modular and maintainable Python codebase

Dedicated scripts for model training and prediction

Full unit test coverage with pytest

Dockerized services for reproducible and environment-independent execution

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Unit Testing](#unit-testing)
- [Dockerized services](#dockerized-services)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Project Structure](#project-structure)
---

## Project Overview

This project uses machine learning models (XGBoost) to predict the total number of bikes rented in a given hour or day. It explores and engineers features like:

- Weather: weathersit, temperature, humidity, windspeed
- Time-based: season, month, hour, weekday, working day

---

## Dataset

The data is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). Place the dataset files in a `data/` folder.

```
data/
├── hour.csv   # Hourly data
```
---

## Setup

### 1. **Create a Conda Environment**

```bash
conda create -n myenv python=3.11.11
conda activate myenv
```

---

### 2. **Install Requirements**

### ➤ For Prediction Only

Install only the minimal dependencies to run the trained model:

```bash
pip install -r requirements/requirements.txt
```
---

### ➤ For Full Development

Includes all tools needed for development: training, testing, Jupyter notebooks, linting, formatting, and Git hooks.

Install the development dependencies:

```bash
pip install -r requirements/requirements-dev.txt
```

#### Updating Dependency Files

If you've changed `requirements.in` or `requirements-dev.in`, use [`pip-compile`](https://github.com/jazzband/pip-tools) to regenerate the pinned `.txt` files:

```bash
pip install pip-tools  # if not already installed

# Compile base dependencies
pip-compile requirements/requirements.in

# Compile development dependencies
pip-compile requirements/requirements-dev.in
```

This ensures all dependencies (and their sub-dependencies) are locked with specific versions for reproducibility.

---

### Verify Installation

Make sure everything was installed correctly:

```bash
python -m pytest --version
pre-commit --version
```

---

### Set Up Pre-commit Hooks

To automatically run formatting and lint checks before each commit:

```bash
pre-commit install
```

(Optional) To run checks manually on all files:

```bash
pre-commit run --all-files
```
---

## Running the Project

### Train the Model

```bash
python src/train.py
```

This script handles:
- Data preprocessing
- Feature engineering
- Model training
- Saving the model to `models/`

### Predict on Test Data

```bash
python predict.py
```
This script handles:
- Data preprocessing
- Feature engineering
- Prediction
- Saving the results to `data/prediction/test`

### Predict on New Data

```bash
python predict.py --daily_infer True
```
The results saved to `data/prediction/infer`

The script will load the trained model and generate daily prediction based on the input features.

---

## Unit Testing

Basic unit tests are included to validate core functionalities such as:

- Data preprocessing
- Feature transformations
- Model loading and prediction

### Run All Tests

```bash
pytest tests/
```

Tests are located in the `/tests` directory and follow the naming convention `test_*.py`.

Example test file:

```
tests/
└── test_preprocessing.py
```
---

## Dockerized services

This project provides Docker-based services to:
- Train a machine learning model
- Run daily predictions

Each service automatically runs unit tests (using `pytest`) **before** executing the main script, ensuring that only passing code is deployed.


### Build Docker Images

To build the Docker images defined in your `docker-compose.yml`:
```bash
docker compose build
```
To rebuild without cache (clean install):
```bash
docker compose build --no-cache
```

### Start the Full Application
To build and start all services defined in the compose file:
```bash
docker compose up
```

### Stop and Clean Up
```bash
docker compose down
```

### Run Training
Runs all tests and then starts training:
```bash
docker compose run --rm train-model
```
If tests fail, training will not start.

### Run Prediction
Runs all tests and then executes the prediction script:
```bash
docker compose run --rm daily-predict
```

### Environment Variables
The following environment variables are managed via the .env file:
```
VERSION=0.1.0
ENVIRONMENT=dev
```
---

## Jupyter Notebooks

You can explore the EDA(Exploratory Data Analysis) in a Jupyter Notebook (development mode, requirements-dev.txt needed to be installed):

```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## Project Structure

```
bike-rent-prediction/
├── data/
│    ├── hour.csv           # Raw dataset files (hour.csv)
│    ├── prediction         # Prediction results
│    ├── ├── ..
├── models/                 # Trained models (e.g. model.json)
├── notebooks/              # Jupyter notebooks (data exploratary analysis)
│   ├── EDA.py
    ..
├── reports/                # Evaluation metrics, plots
├── src/                    # Source code (main python modules)
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning & feature engineering
│   ├── train.py            # Model training logic
│   ├── predict.py          # Prediction logic
│   └── utils/              # Helper functions folder
│   └── └── ..
├── tests/                  # Unit tests (test_*.py)
│   └── test_preprocessing.py
│   └── test_train.py
│   └── test_prediction.py
├── requirements/
│    ├── requirements.txt
│    ├── requirements-dev.txt
│    ..
├── Dockerfile
├── docker-compose.yaml
├── .env.template
└── README.md

```
