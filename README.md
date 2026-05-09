# Amazon Fashion Recommendation System

A collaborative filtering(both User-based and Item-based), content-based and hybrid recommendation system built on the [Amazon Reviews (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) dataset (`AMAZON_FASHION_5.jsonl`). Includes a Flask API and a static web UI.

---

## Methods

| Method | Description |
|---|---|
| User-Based CF | Finds similar users and predicts ratings based on their preferences |
| Item-Based CF | Finds similar items to those a user has already rated |
| Content-Based | Matches items based on style metadata (color, size, etc.) |
| Hybrid | Weighted combination of all three (40% UCF + 40% ICF + 20% CB) |

**Similarity metrics:** cosine, Pearson, Euclidean

**Prediction strategies:** weighted average, mean-centered, significance weighting

---

## Project Structure

```
.
├── recommender.py          # Core algorithms, similarity/prediction functions, evaluation
├── app.py                  # Flask REST API
├── get_recommendations.py  # Quick CLI demo
└── static/                 # Web UI (served by Flask)
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install flask
```

Download the dataset from [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and place `AMAZON_FASHION_5.jsonl` in the project root.

---

## Usage

### CLI demo

```bash
python get_recommendations.py
```

### Run experiments (full evaluation)

```bash
python recommender.py AMAZON_FASHION_5.jsonl
```

This runs a train/test split (80/20) and prints RMSE, MAE, Precision@10, Recall@10, F1@10, and Coverage across all method combinations.

### Flask API

```bash
python app.py
# or
DATA_FILE=AMAZON_FASHION_5.jsonl python app.py
```

Server starts at `http://localhost:5000`.

---

## API Endpoints

**GET /api/users**
Returns all users sorted by number of ratings.

**GET /api/user/`<user_id>`/history**
Returns a user's rating history.

**GET /api/recommend**

| Param | Options | Default |
|---|---|---|
| `user_id` | any valid user ID | required |
| `method` | `user_cf`, `item_cf`, `content`, `hybrid` | `hybrid` |
| `similarity` | `cosine`, `pearson`, `euclidean` | `cosine` |
| `prediction` | `weighted_average`, `mean_centered`, `significance_weighting` | `mean_centered` |
| `n` | integer | `5` |

Example:
```
GET /api/recommend?user_id=A3BN0MRGRDKM0J&method=hybrid&n=5
```

Each recommendation includes the predicted score, method used, a plain-English reason, and item metadata.

---

## Dataset

Jianmo Ni, Jiacheng Li, Julian McAuley. *Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects.* EMNLP 2019.
