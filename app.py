import json
import os
import sys
from collections import defaultdict
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from recommender import (
    load_data, build_matrices,
    UserBasedCF, ItemBasedCF, ContentBasedFilter, HybridRecommender,
)

DATA_FILE = os.environ.get("DATA_FILE", "AMAZON_FASHION_5.jsonl")

app = Flask(__name__, static_folder="static")

print(f"[boot] Loading {DATA_FILE} …")
records = load_data(DATA_FILE)
user_item, item_user, item_meta, user_means, item_means = build_matrices(records)

reviewer_names: dict[str, str] = {}
reviewer_ratings_count: dict[str, int] = defaultdict(int)
for rec in records:
    uid = rec.get("reviewerID", "")
    if uid and rec.get("reviewerName"):
        reviewer_names[uid] = rec["reviewerName"]
    if uid:
        reviewer_ratings_count[uid] += 1

item_styles: dict[str, dict] = defaultdict(dict)
for rec in records:
    iid = rec.get("asin", "")
    style = rec.get("style", {})
    if iid and style:
        for k, v in style.items():
            item_styles[iid][k.strip().rstrip(":")] = str(v).strip()

print(f"[boot] Ready — {len(user_item)} users, {len(item_user)} items")



def build_models(similarity: str, prediction: str, top_k: int = 20):
    ucf = UserBasedCF(user_item, user_means,
                      similarity=similarity, prediction=prediction, top_k=top_k)
    icf = ItemBasedCF(item_user, user_item, item_means, user_means,
                      similarity=similarity, prediction=prediction, top_k=top_k)
    cb  = ContentBasedFilter(user_item, item_meta)
    hybrid = HybridRecommender(ucf, icf, cb, w_user=0.4, w_item=0.4, w_content=0.2)
    return {"user_cf": ucf, "item_cf": icf, "content": cb, "hybrid": hybrid}


def exp_to_dict(exp) -> dict:
    """Serialise an Explanation object to a JSON-safe dict."""
    return {
        "item_id": exp.item_id,
        "score":   round(exp.score, 4),
        "method":  exp.method,
        "reason":  exp.reason,
        "details": exp.details,
        "style":   item_styles.get(exp.item_id, {}),
        "avg_rating": round(item_means.get(exp.item_id, 0), 2),
        "num_ratings": len(item_user.get(exp.item_id, {})),
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/users")
def api_users():
    """Return all users with name + rating count, sorted by rating count."""
    users = []
    for uid in user_item:
        users.append({
            "id":           uid,
            "name":         reviewer_names.get(uid, uid[:12] + "…"),
            "num_ratings":  reviewer_ratings_count[uid],
            "avg_rating":   round(user_means.get(uid, 0), 2),
        })
    users.sort(key=lambda u: -u["num_ratings"])
    return jsonify(users)


@app.route("/api/user/<user_id>/history")
def api_user_history(user_id):
    """Return the rating history for a given user."""
    ratings = user_item.get(user_id, {})
    history = []
    for iid, r in ratings.items():
        history.append({
            "item_id":    iid,
            "rating":     r,
            "style":      item_styles.get(iid, {}),
            "avg_rating": round(item_means.get(iid, 0), 2),
        })
    history.sort(key=lambda x: -x["rating"])
    return jsonify(history)


@app.route("/api/recommend")
def api_recommend():
    """
    Query params:
      user_id    : required
      method     : user_cf | item_cf | content | hybrid  (default: hybrid)
      similarity : cosine | pearson | euclidean           (default: cosine)
      prediction : weighted_average | mean_centered | significance_weighting
                                                          (default: mean_centered)
      n          : number of recommendations              (default: 5)
    """
    user_id    = request.args.get("user_id", "")
    method     = request.args.get("method", "hybrid")
    similarity = request.args.get("similarity", "cosine")
    prediction = request.args.get("prediction", "mean_centered")
    n          = int(request.args.get("n", 5))

    if not user_id or user_id not in user_item:
        return jsonify({"error": "Unknown user_id"}), 400

    valid_sim  = {"cosine", "pearson", "euclidean"}
    valid_pred = {"weighted_average", "mean_centered", "significance_weighting"}
    if similarity not in valid_sim:
        return jsonify({"error": f"similarity must be one of {valid_sim}"}), 400
    if prediction not in valid_pred:
        return jsonify({"error": f"prediction must be one of {valid_pred}"}), 400

    models = build_models(similarity, prediction)
    model  = models.get(method)
    if model is None:
        return jsonify({"error": f"Unknown method '{method}'"}), 400

    try:
        explanations = model.recommend_with_explanations(user_id, n=n)
        return jsonify([exp_to_dict(e) for e in explanations])
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
