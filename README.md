### TECHNICAL TEST RECOMMENDER SYSTEM
# 🧠 Implicit Feedback Recommender System

This project implements a collaborative filtering recommender system based on implicit feedback, using techniques like Alternating Least Squares (ALS), leave-one-out evaluation. It is designed to work efficiently on large-scale transactional data with sparse user-item interactions.

## 📂 Project Structure

- `dataset/collaborative-filtering-data` – processed data for matrix factorizatoion 
- `dataset/datas` -- Raw datas
- `./` – notebooks for development and exploratory notebooks for preprocessing, model training, evaluation
- `program-v1/` – Gradio interface that run the model
- `program-v1/weights` -- store weights of different experimentation
- `models/` – Saved ALS models with unique timestamp postfixes

## 🔍 Features

- Efficient interaction matrix generation using `csr_matrix`
- Leave-one-out evaluation for model performance tracking
- Hit Rate@K metric implementation
- Filtering and merging tools for handling large transactional datasets
- Utility tools for tracking time, saving models, etc.

## 📊 Evaluation Metrics

### Hit Rate@K
Measures the proportion of users for whom the true next item appears in the top-K recommended items.  
- Bounded between 0 and 1
- Easy to interpret
- Best used alongside metrics like NDCG@K

## 🚀 Getting Started

> pre-process and generate the data using `dataprep-collaborative-filtering.ipynb` or using the `program-v1/ALS_dataprep_lib.py`

> run the `program-v1/ALS_training.py` to get weights

> run the `program-v1/gradio_interface.py` and enter a customer_id

