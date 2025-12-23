# Heart Disease Prediction Pipeline

This repository implements an end-to-end **Machine Learning pipeline** for predicting heart disease using the **UCI Heart Disease (Cleveland) dataset**. It covers data acquisition, preprocessing, model training, evaluation, API deployment, containerization, Kubernetes deployment, and monitoring.

---

## Dataset

* **Source:** UCI Heart Disease Dataset (Cleveland subset)
* **Features:** 14 clinical attributes
* **Target Variable:**

  * `0` → No heart disease
  * `1` → Presence of heart disease

---

## Task 1: Data Acquisition & Exploratory Data Analysis (EDA)

### Data Acquisition

Script used:

```bash
python data/download_scripts/download_ucidata.py
```

* Downloads the dataset programmatically
* Ensures reproducibility

### Data Cleaning

* Replace `?` with `NaN`
* Remove rows with missing values
* Binarize the target column:

  * `0` → No heart disease
  * `1` → Presence of heart disease

### Exploratory Data Analysis

* Histograms
* Correlation heatmaps
* Class balance plots

**Key Observations:**

* Mild class imbalance
* Strong correlation of `thalach`, `oldpeak`, and `exang` with the target

All plots are saved under:

```text
screenshots/
```

---

## Task 2: Data Preprocessing

Preprocessing script:

```text
src/preprocess.py
```

### Operations Performed

* Encode categorical variables
* Scale numerical features using `StandardScaler`
* Save preprocessing pipeline for inference

### Example Usage

```bash
python src/preprocess.py \
  --input data/heart_cleveland.csv \
  --output processed/processed_data.csv
```

---

## Task 3: Model Training

### Training Scripts

* `src/train.py` – Baseline model
* `src/train_final_model.py` – Final model (RandomForest + preprocessing pipeline)

### Output

* Trained model saved as:

```text
models/heart_disease_model.pkl
```

### Run Final Training

```bash
python src/train_final_model.py
```

* Training metrics are printed to the console

---

## Task 4: Model Evaluation

Evaluate the model using the test set:

```bash
python src/test_model.py
```

### Evaluation Metrics

* Accuracy
* F1-score
* ROC-AUC

Confusion matrix plots are saved to:

```text
screenshots/
```

---

## Task 5: Data Loading & Testing

Ensure reproducibility of preprocessing and test data loading:

```bash
python src/test_data_loading.py
```

* Validates the preprocessing pipeline
* Confirms compatibility with the test dataset

---

## Task 6: Model Containerization (API)

### API Details

* Framework: **FastAPI**
* Endpoint: `/predict`
* Input: JSON
* Output:

```json
{
  "prediction": 0,
  "probability": 0.255
}
```

### Build Docker Image

```bash
docker build -t heart-disease-api -f docker/Dockerfile .
```

### Run Container Locally

```bash
docker run -p 8000:8000 heart-disease-api
```

### Test API Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'
```

---

## Task 7: Production Deployment (Kubernetes)

### Kubernetes Manifests

* `k8s/deployment.yaml`
* `k8s/service.yaml`

### Apply Deployment

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Verify Deployment

```bash
kubectl get pods
kubectl get svc
```

* Access the API using **LoadBalancer IP** or **NodePort**
* Capture screenshots of the deployment for submission

---

## Task 8: Monitoring & Logging

### Logging

* Implemented in `src/app.py`
* Logs include:

  * API requests
  * Predictions
  * Errors

Logs are stored inside the container at:

```text
/app/logs/api.log
```

### Optional Monitoring

* Prometheus + Grafana
* Library: `prometheus_fastapi_instrumentator`
* Metrics endpoint:

```text
/metrics
```

### Check Logs in Running Container

```bash
docker exec -it <container_id> tail -f /app/logs/api.log
```

### Example Log Entry

```text
2025-12-23 14:30:25 | INFO | Received prediction request: {...}
2025-12-23 14:30:25 | INFO | Prediction result: prediction=0, probability=0.255
```

---

## Summary

This project demonstrates a complete ML lifecycle:

* Reproducible data ingestion
* Clean preprocessing pipeline
* Model training and evaluation
* Production-ready API
* Containerization and Kubernetes deployment
* Logging and monitoring
