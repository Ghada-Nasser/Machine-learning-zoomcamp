# Machine Learning Zoomcamp – My Learning Journey

This repository documents my learning journey through the **[Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)** by DataTalksClub.

The course provides hands-on experience in building machine learning systems, from understanding the problem and exploring data to training, evaluating, and deploying models.

This repository contains my notes, homework assignments, practice notebooks, and projects as I progress through the course.

## Course Progress

<badge color="success">Completed through Midterm Project</badge>

### Module 1: Introduction to Machine Learning

Learned the fundamentals of machine learning and how to approach ML problems systematically.

**Topics covered:**
- Machine learning vs. rule-based systems.
- Supervised learning fundamentals.
- CRISP-DM methodology.
- Problem framing and model selection.
- Development environment setup.

**Key takeaway:** Understanding how to structure a machine learning project and approach problems systematically.

[Module 1 Materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/01-intro)

### Module 2: Machine Learning for Regression

Built a foundation in regression by working on a car-price prediction problem.

**Topics covered:**
- Exploratory Data Analysis (EDA).
- Data preparation and feature engineering.
- Linear regression from scratch and with Scikit-learn.
- Feature selection.
- Regularization.
- Model validation and evaluation.

**Key takeaway:** Building regression models through a structured workflow, from data exploration to model evaluation.

[Module 2 Materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/02-regression)

### Module 3: Machine Learning for Classification

Learned classification techniques through a customer churn prediction problem.

**Topics covered:**
- Logistic Regression.
- Categorical feature encoding.
- Feature importance.
- Feature selection.
- Understanding and interpreting model predictions.

**Key takeaway:** Building classification models for predicting categorical outcomes using structured data.

[Module 3 Materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/03-classification)

### Module 4: Evaluation Metrics for Classification

Focused on evaluating classification models and understanding how to select appropriate metrics.

**Topics covered:**
- Accuracy, Precision, Recall, and F1 Score.
- Confusion Matrix.
- ROC Curves and AUC.
- Cross-validation.
- Evaluation with imbalanced datasets.

**Key takeaway:** Evaluating models beyond accuracy and selecting metrics based on the actual problem requirements.

[Module 4 Materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/04-evaluation)

### Module 5: Deploying Machine Learning Models

Learned how to turn a trained machine learning model into a service that can be accessed through an API.

**Topics covered:**
- Model serialization.
- Building prediction APIs with FastAPI.
- Docker fundamentals.
- Containerizing machine learning applications.
- Introduction to cloud deployment.

**Key takeaway:** Understanding the steps required to make a trained model accessible as a web service.

[Module 5 Materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/05-deployment)

### Module 6: Decision Trees and Ensemble Learning

Explored tree-based algorithms and ensemble methods to improve predictive performance.

**Topics covered:**
- Decision Trees.
- Random Forest.
- Gradient Boosting with XGBoost.
- Hyperparameter tuning.
- Feature importance analysis.

**Key takeaway:** Training and tuning tree-based models and comparing their performance with other machine learning algorithms.

[Module 6 Materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/06-trees)

---

## Midterm Project: Online Shoppers Purchasing Intention Prediction

As part of the midterm project, I applied the concepts learned in Modules 1–6 to an end-to-end machine learning problem.

**Project Goal:**

Predict whether an online shopping session will result in a purchase based on visitor behavior and session characteristics.

**Dataset:** [Online Shoppers Purchasing Intention Dataset – UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

**What I implemented:**
- Exploratory Data Analysis and data preprocessing.
- Training and comparison of multiple classification models.
- Model evaluation using classification metrics.
- Hyperparameter and classification threshold experiments.
- Selection of a Random Forest model.
- Model serialization using Pickle.
- Building a prediction API using Flask.
- Containerizing the application using Docker.

**Technologies used:**
- Python, Pandas, NumPy.
- Scikit-learn.
- Matplotlib and Seaborn.
- Flask.
- Docker.

**Project repository:** [Online Shoppers Purchasing Intention Prediction](./midterm_project)

The project brings together the main concepts from the first six modules into a practical machine learning application.

## Repository Structure

```text
Machine-learning-zoomcamp/
│
├── Homework1.ipynb
├── Homework2.ipynb
├── Homework3.ipynb
├── Homework4.ipynb
├── Homework5/
├── Homework6/
│
└── midterm_project/
    ├── data_exploration.ipynb
    ├── train.py
    ├── predict.py
    ├── app.py
    ├── models/
    ├── Dockerfile
    ├── pyproject.toml
    └── README.md
```

*Note: The repository structure may evolve as I continue through the course.*

---

**Course:** [Machine Learning Zoomcamp – DataTalksClub](https://github.com/DataTalksClub/machine-learning-zoomcamp)

**Status:** Completed Modules 1–6 and Midterm Project.
