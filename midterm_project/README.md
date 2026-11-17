# Online Shoppers Purchasing Intention Prediction

## 1. Project Overview

### Problem Description

E-commerce websites receive many visitors, but not every visitor completes a purchase. Understanding which browsing sessions are more likely to result in a purchase can help businesses improve their marketing strategies, personalize the shopping experience, and allocate resources more effectively.

The goal of this project is to build a machine learning model that predicts whether an online shopping session will result in a purchase (`Revenue = True`) or not (`Revenue = False`), based on visitors' browsing behavior and session characteristics.

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

Each row represents a browsing session, and the target variable is `Revenue`.

### Project Workflow

The project follows an end-to-end machine learning workflow:

1. Data collection
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Model training and evaluation
5. Model selection and threshold tuning
6. Model serialization
7. API development using Flask
8. Docker containerization

## 2. Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)


## 3. Exploratory Data Analysis (EDA)

The EDA is available in:

`data_exploration.ipynb`


## 4. Project Structure

```text
midterm_project/
│
├── models/
│   ├── model.pkl
│   ├── dv.pkl
│   └── threshold.pkl
│
├── data_exploration.ipynb
├── train.py
├── predict.py
├── app.py
├── pyproject.toml
├── Dockerfile
└── README.md
```

## 5. Environment Setup and Installation

### Requirements

- Python 3.10 or higher
- pip
- Docker (for containerized execution)

### Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd midterm_project
```

### Create a Virtual Environment

```bash
python -m venv midterm-project
```

### Activate the Environment

On Linux/macOS:

```bash
source midterm-project/bin/activate
```

On Windows:

```bash
midterm-project\Scripts\activate
```

### Install Dependencies

The project dependencies are defined in `pyproject.toml`.

```bash
pip install .
```

## 6. Training the Model

To download the dataset, preprocess the data, train the Random Forest model, evaluate its performance, and save the model artifacts, run:

```bash
python train.py
```

The trained model and preprocessing artifacts will be saved in the `models/` directory.

## 7. Running the Flask API

Start the Flask application:

```bash
python app.py
```

The API will run on:

`http://localhost:5000`

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict purchase intention |

### Prediction Request Example

Send a POST request to:

`http://localhost:5000/predict`

Example JSON request:

```json
{
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 10,
    "ProductRelated_Duration": 300.0,
    "BounceRates": 0.02,
    "ExitRates": 0.05,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": "May",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
}
```

### Prediction Response Example

```json
{
    "prediction": false,
    "purchase_probability": 0.445
}
```

The `purchase_probability` represents the estimated probability of a purchase.

The final prediction is `true` when the purchase probability is greater than or equal to the selected threshold (0.6), and `false` otherwise.

## 8. Docker Containerization

The project includes a `Dockerfile` to package the Flask API, its dependencies, and the trained model artifacts into a Docker image.

### Build the Docker Image

Run the following command from the project directory:

```bash
docker build -t online-shoppers-prediction .
```

### Run the Container

```bash
docker run -d \
    --name online-shoppers-api \
    -p 5000:5000 \
    online-shoppers-prediction
```

The API will be accessible at:

`http://localhost:5000`

### Check the Running Container

```bash
docker ps
```

### Test the API Health

```bash
curl http://localhost:5000/health
```

Expected response:

```json
{
    "status": "ok"
}
```

### Stop the Container

```bash
docker stop online-shoppers-api
```

### Remove the Container

```bash
docker rm online-shoppers-api
```

## 9. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- UCI Machine Learning Repository
- Flask
- Pickle
- Docker
