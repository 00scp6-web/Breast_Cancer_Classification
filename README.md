# Breast Cancer Classification: A Comparative Study of Ensemble Learning Techniques

## Project Overview

This project compares four machine-learning classification models using the Breast Cancer Wisconsin Diagnostic dataset. An interactive Streamlit dashboard displays model performance, predictions, and visualizations.

## Objectives

* Train and compare four classification models.
* Evaluate model performance using standard classification metrics.
* Visualize model results through an interactive dashboard.
* Explore predictions and feature information.

## Dataset

The dataset is provided by scikit-learn and contains:

* 569 samples
* 30 numerical features
* Malignant and benign target classes

Target labels:

* 0: Malignant
* 1: Benign

## Machine-Learning Models

1. Logistic Regression
2. Random Forest
3. AdaBoost
4. Gradient Boosting

## Technologies Used

* Python
* Streamlit
* NumPy
* pandas
* Matplotlib
* scikit-learn

## Project Structure

* `app.py` — Streamlit dashboard
* `main.py` — Model training and evaluation
* `individual_models/` — Individual model scripts
* `results/` — Model comparison results and graphs
* `requirements.txt` — Required Python libraries

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Generate the evaluation results:

```bash
python main.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Results

In the recorded experiment, Logistic Regression achieved 98.25% accuracy and 97.62% malignant recall. The other models can be compared using the dashboard's performance analysis.

## Disclaimer

This project is for educational purposes only. It is not a medical diagnostic tool and must not be used to make clinical decisions.
