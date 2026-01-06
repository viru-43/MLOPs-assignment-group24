# 🫀 Heart Disease Prediction – End-to-End MLOps Project

This repository contains an **end-to-end MLOps pipeline** for predicting heart disease using the **UCI Heart Disease Dataset**.  
The project demonstrates modern MLOps best practices including **experiment tracking, CI/CD, containerization, Kubernetes deployment, and monitoring**.

---

## 📌 Project Overview

The objective of this project is to build a machine learning classifier that predicts the **risk of heart disease** based on patient medical attributes and deploy it as a **production-ready API**.

The complete lifecycle covered in this project includes:
- Data acquisition and EDA
- Model training and evaluation
- Experiment tracking using MLflow
- API development using FastAPI
- Docker containerization
- CI/CD pipeline using GitHub Actions
- Kubernetes deployment (Docker Desktop)
- Monitoring and logging

---

## 📊 Dataset

- **Dataset Name:** Heart Disease UCI Dataset  
- **Source:** UCI Machine Learning Repository  
- **Samples:** 303  
- **Features:** 13 medical attributes  
- **Target:** Binary classification  
  - `0` → No heart disease  
  - `1` → Presence of heart disease  

The dataset is programmatically fetched using the `ucimlrepo` library and saved locally for reproducibility.

---

## 🧱 System Architecture

The project follows a layered MLOps architecture:

- **Data Layer:** UCI dataset, EDA using Pandas and Jupyter  
- **Training Layer:** Scikit-learn pipeline with preprocessing and Logistic Regression  
- **Experiment Tracking:** MLflow for metrics and model artifacts  
- **Serving Layer:** FastAPI REST API (`/predict`, `/metrics`)  
- **Containerization:** Docker  
- **Deployment:** Kubernetes (Docker Desktop, NodePort service)  
- **CI/CD:** GitHub Actions  
- **Monitoring:** API logs and metrics endpoint  

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- Git
- Docker Desktop (with Kubernetes enabled)
- kubectl

---

### Clone Repository

```bash
git clone <your-github-repo-url>
cd mlops-assignment
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Data Acquisition

Fetch and prepare the dataset:

```bash
python src/data/load_data.py
```

The cleaned dataset will be saved to:

```
data/raw/heart.csv
```

---

## 📈 Exploratory Data Analysis

EDA is performed in the notebook:

```
notebooks/01_eda.ipynb
```

It includes:
- Target distribution
- Feature histograms
- Correlation heatmap

---

## 🤖 Model Training & Experiment Tracking

Train the model and log experiments using MLflow:

```bash
python src/models/train.py
```

Launch MLflow UI:

```bash
mlflow ui
```

Access at:
```
http://127.0.0.1:5000
```

---

## 🚀 Running the API Locally

Start FastAPI server:

```bash
uvicorn src.api.app:app --reload
```

Swagger UI:
```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker Containerization

### Build Docker Image

```bash
docker build -t heart-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 heart-api
```

Access API:
```
http://127.0.0.1:8000/docs
```

---

## ☸️ Kubernetes Deployment (Local)

### Enable Kubernetes
- Open Docker Desktop
- Settings → Kubernetes → Enable

---

### Deploy to Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Verify:

```bash
kubectl get pods
kubectl get services
```

---

### Access Deployed API

```
http://localhost:30007/docs
```

---

## 📊 Monitoring & Logging

- API request and prediction logs are printed to the console
- A `/metrics` endpoint provides basic monitoring information

Metrics endpoint:

```
http://localhost:30007/metrics
```

---

## 🔄 CI/CD Pipeline

A CI pipeline is implemented using **GitHub Actions**:
- Installs dependencies
- Runs automated tests using Pytest
- Triggered on every push to the repository

Pipeline status can be viewed in the **Actions** tab on GitHub.

---

## 📁 Repository Structure

```
mlops-assignment/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── src/
│   ├── api/
│   ├── data/
│   └── models/
├── tests/
├── k8s/
├── .github/workflows/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📄 Deliverables

- Complete GitHub repository
- Architecture diagram
- Screenshots for all stages
- Final report (DOCX/PDF)
- Short demonstration video

---

## ✅ Conclusion

This project demonstrates a **complete production-ready MLOps workflow**, covering the full lifecycle of a machine learning system from data ingestion to Kubernetes deployment and monitoring.  
The implementation aligns with industry best practices and academic requirements.

---

## 📬 Author

Group 24:
1.	Virendra Vinay Sahakari (2024AB05036)
2.	Snehal Swapnil Chavan (2024AB05033) 
3.	Shripad Prakash Kelapure (2024AA05957)
4.	Abhishek Kumar (2024AB05032) 
**Course:** MLOps (S1-25_AIMLCZG523)  
**Assignment:** Experimental Learning Assignment – I  
