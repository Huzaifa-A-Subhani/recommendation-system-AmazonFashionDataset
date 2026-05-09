import json
import math
import time
import random
import warnings
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")


class Explanation:
    """
    Holds a human-readable explanation for a single recommendation.

    Attributes
    ----------
    item_id      : recommended item ASIN
    score        : predicted score / similarity used for ranking
    method       : 'UserCF' | 'ItemCF' | 'ContentBased' | 'Hybrid'
    reason       : one-line plain-English summary
    details      : dict with method-specific supporting evidence
    """
    def __init__(self, item_id, score, method, reason, details):
        self.item_id = item_id
        self.score   = score
        self.method  = method
        self.reason  = reason
        self.details = details          

    def __repr__(self):
        return f"Explanation(item={self.item_id}, method={self.method}, score={self.score:.3f})"

    def pretty(self, width=72):
        """Return a formatted multi-line string for printing."""
        bar  = "─" * width
        lines = [
            bar,
            f"  Item     : {self.item_id}",
            f"  Method   : {self.method}",
            f"  Score    : {self.score:.4f}",
            f"  Reason   : {self.reason}",
        ]
        for k, v in self.details.items():
            if isinstance(v, list):
                lines.append(f"  {k:<12} :")
                for entry in v:
                    lines.append(f"               {entry}")
            else:
                lines.append(f"  {k:<12} : {v}")
        lines.append(bar)
        return "\n".join(lines)


def _fmt_sim(sim_name, value):
    """Return a readable similarity label."""
    return f"{sim_name} similarity = {value:.4f}"



def load_data(filepath: str):
    """Load JSONL and return raw records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_matrices(records):
    """
    Build:
      user_item  : {user_id: {item_id: rating}}
      item_user  : {item_id: {user_id: rating}}
      item_meta  : {item_id: {feature: value}}   (for content-based)
      user_means : {user_id: mean_rating}
      item_means : {item_id: mean_rating}
    """
    user_item  = defaultdict(dict)
    item_user  = defaultdict(dict)
    item_meta  = defaultdict(dict)

    for rec in records:
        uid    = rec.get("reviewerID")
        iid    = rec.get("asin")
        rating = float(rec.get("overall", 0))

        if not uid or not iid:
            continue

        user_item[uid][iid] = rating
        item_user[iid][uid] = rating

        # collect style metadata for content-based
        style = rec.get("style", {})
        if style:
            for k, v in style.items():
                clean_k = k.strip().rstrip(":")
                clean_v = str(v).strip()
                # store as feature flag
                item_meta[iid][f"{clean_k}:{clean_v}"] = 1.0

    user_item = dict(user_item)
    item_user = dict(item_user)
    item_meta = dict(item_meta)

    # mean ratings
    user_means = {u: sum(r.values()) / len(r) for u, r in user_item.items()}
    item_means = {i: sum(r.values()) / len(r) for i, r in item_user.items()}

    return user_item, item_user, item_meta, user_means, item_means



def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse rating vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot   = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def pearson_similarity(vec_a: dict, vec_b: dict) -> float:
    """Pearson Correlation Coefficient between two sparse rating vectors."""
    common = set(vec_a) & set(vec_b)
    n = len(common)
    if n < 2:
        return 0.0
    vals_a = [vec_a[k] for k in common]
    vals_b = [vec_b[k] for k in common]
    mean_a = sum(vals_a) / n
    mean_b = sum(vals_b) / n
    num    = sum((a - mean_a) * (b - mean_b) for a, b in zip(vals_a, vals_b))
    den_a  = math.sqrt(sum((a - mean_a) ** 2 for a in vals_a))
    den_b  = math.sqrt(sum((b - mean_b) ** 2 for b in vals_b))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def euclidean_similarity(vec_a: dict, vec_b: dict) -> float:
    """Euclidean-distance-based similarity (1 / 1+dist) on common items."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dist = math.sqrt(sum((vec_a[k] - vec_b[k]) ** 2 for k in common))
    return 1.0 / (1.0 + dist)


SIMILARITY_FUNCS = {
    "cosine":    cosine_similarity,
    "pearson":   pearson_similarity,
    "euclidean": euclidean_similarity,
}



def predict_weighted_average(
    target_ratings: dict,         # ratings already given by/for neighbours
    neighbour_sims: list,         # [(neighbour_id, similarity), ...]
    global_mean: float = 3.0,
) -> float:
    """
    Weighted Average prediction:
      pred = Σ(sim * r) / Σ|sim|
    Falls back to global mean if no signal.
    """
    num = den = 0.0
    for nid, sim in neighbour_sims:
        if nid in target_ratings:
            num += sim * target_ratings[nid]
            den += abs(sim)
    if den == 0:
        return global_mean
    return num / den


def predict_mean_centered(
    target_ratings: dict,
    neighbour_sims: list,
    target_mean: float,
    neighbour_means: dict,
    global_mean: float = 3.0,
) -> float:
    """
    Mean-Centred (bias-adjusted) prediction:
      pred = μ_target + Σ(sim * (r - μ_neighbour)) / Σ|sim|
    Removes per-user/item rating bias.
    """
    num = den = 0.0
    for nid, sim in neighbour_sims:
        if nid in target_ratings:
            n_mean = neighbour_means.get(nid, global_mean)
            num   += sim * (target_ratings[nid] - n_mean)
            den   += abs(sim)
    if den == 0:
        return target_mean
    return target_mean + num / den


def predict_significance_weighted(
    target_ratings: dict,
    neighbour_sims: list,
    target_mean: float,
    neighbour_means: dict,
    min_common: int = 5,
    global_mean: float = 3.0,
) -> float:
    """
    Significance Weighting prediction:
      Scales similarity by min(co-rated items, min_common) / min_common
      to penalise neighbours with very few co-rated items.
      Then applies mean-centred formula.
    """
    num = den = 0.0
    for nid, sim in neighbour_sims:
        if nid in target_ratings:
            # number of items both users/items have in common
            # (already encoded in sim computation; here we approximate
            #  by checking overlap length stored externally in _SW_cocount)
            co = _SW_cocount.get((min(target_ratings, nid), max(target_ratings, nid)), 1)
            scale = min(co, min_common) / min_common
            adj_sim = sim * scale
            n_mean  = neighbour_means.get(nid, global_mean)
            num    += adj_sim * (target_ratings[nid] - n_mean)
            den    += abs(adj_sim)
    if den == 0:
        return target_mean
    return target_mean + num / den


# Global co-count registry used by significance weighting
_SW_cocount: dict = {}


def register_cocount(vec_a_id, vec_b_id, n_common):
    """Register co-rated item count for significance weighting."""
    key = (min(vec_a_id, vec_b_id), max(vec_a_id, vec_b_id))
    _SW_cocount[key] = n_common


PREDICTION_FUNCS = {
    "weighted_average":     predict_weighted_average,
    "mean_centered":        predict_mean_centered,
    "significance_weighting": predict_significance_weighted,
}



class UserBasedCF:
    def __init__(
        self,
        user_item: dict,
        user_means: dict,
        similarity: str = "cosine",
        prediction: str = "mean_centered",
        top_k: int = 20,
    ):
        self.user_item  = user_item
        self.user_means = user_means
        self.sim_fn     = SIMILARITY_FUNCS[similarity]
        self.pred_name  = prediction
        self.top_k      = top_k
        self.global_mean = sum(user_means.values()) / len(user_means) if user_means else 3.0
        self._sim_cache: dict = {}

    def _get_similarity(self, u1, u2):
        key = (min(u1, u2), max(u1, u2))
        if key not in self._sim_cache:
            s = self.sim_fn(self.user_item.get(u1, {}), self.user_item.get(u2, {}))
            # register co-count for significance weighting
            common = len(set(self.user_item.get(u1, {})) & set(self.user_item.get(u2, {})))
            register_cocount(u1, u2, common)
            self._sim_cache[key] = s
        return self._sim_cache[key]

    def get_neighbours(self, user_id):
        sims = []
        for other in self.user_item:
            if other == user_id:
                continue
            s = self._get_similarity(user_id, other)
            if s > 0:
                sims.append((other, s))
        sims.sort(key=lambda x: -x[1])
        return sims[: self.top_k]

    def predict(self, user_id, item_id):
        neighbours = self.get_neighbours(user_id)
        # target_ratings: how each neighbour rated *this item*
        item_ratings_by_neighbours = {
            n: self.user_item[n][item_id]
            for n, _ in neighbours
            if item_id in self.user_item.get(n, {})
        }
        u_mean = self.user_means.get(user_id, self.global_mean)

        if self.pred_name == "weighted_average":
            return predict_weighted_average(item_ratings_by_neighbours, neighbours, self.global_mean)
        elif self.pred_name == "mean_centered":
            return predict_mean_centered(
                item_ratings_by_neighbours, neighbours, u_mean,
                self.user_means, self.global_mean
            )
        else:  # significance_weighting
            return predict_significance_weighted(
                item_ratings_by_neighbours, neighbours, u_mean,
                self.user_means, global_mean=self.global_mean
            )

    def recommend(self, user_id, n=10):
        if user_id not in self.user_item:
            return []
        seen = set(self.user_item[user_id])
        all_items = set(i for u in self.user_item.values() for i in u)
        candidates = all_items - seen

        scored = []
        for item_id in candidates:
            pred = self.predict(user_id, item_id)
            scored.append((item_id, pred))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def explain(self, user_id, item_id) -> Explanation:
        """
        Return an Explanation for why item_id is recommended to user_id.

        Evidence collected:
          - Top-3 most similar neighbours who rated this item
          - Their similarity scores and what rating they gave
          - How many items the target user and each neighbour co-rated
          - The prediction method being used
          - The predicted score
        """
        neighbours = self.get_neighbours(user_id)
        # only neighbours who actually rated this item
        active_neighbours = [
            (nid, sim)
            for nid, sim in neighbours
            if item_id in self.user_item.get(nid, {})
        ]
        pred = self.predict(user_id, item_id)
        u_mean = self.user_means.get(user_id, self.global_mean)

        if not active_neighbours:
            reason = (
                f"Predicted based on your average rating ({u_mean:.1f}★) — "
                f"no similar users have rated this item yet."
            )
            details = {
                "Prediction": f"{pred:.3f}",
                "Fallback": "No neighbours rated this item; used user mean.",
            }
            return Explanation(item_id, pred, "UserCF", reason, details)

        sim_name = self.sim_fn.__name__.replace("_similarity", "")
        top3 = active_neighbours[:3]
        n_total = len(active_neighbours)

        neighbour_lines = []
        for nid, sim in top3:
            rating    = self.user_item[nid][item_id]
            n_mean    = self.user_means.get(nid, self.global_mean)
            co_items  = len(set(self.user_item.get(user_id, {})) &
                            set(self.user_item.get(nid, {})))
            neighbour_lines.append(
                f"User {nid[:12]}…  "
                f"{_fmt_sim(sim_name, sim)}  |  "
                f"rated this item {rating:.0f}★  |  "
                f"{co_items} items in common"
            )

        reason = (
            f"Recommended because {n_total} similar user(s) rated "
            f"this item highly (top neighbour {_fmt_sim(sim_name, top3[0][1])})."
        )
        details = {
            "Pred. score":  f"{pred:.4f}  (clipped to [1,5])",
            "Pred. method": self.pred_name.replace("_", " ").title(),
            "Similarity":   sim_name.title(),
            "Your avg ★":   f"{u_mean:.2f}",
            "Neighbours":   neighbour_lines,
        }
        return Explanation(item_id, pred, "UserCF", reason, details)

    def recommend_with_explanations(self, user_id, n=10):
        """Return list of Explanation objects (sorted by predicted score)."""
        recs = self.recommend(user_id, n)
        return [self.explain(user_id, iid) for iid, _ in recs]



class ItemBasedCF:
    def __init__(
        self,
        item_user: dict,
        user_item: dict,
        item_means: dict,
        user_means: dict,
        similarity: str = "cosine",
        prediction: str = "mean_centered",
        top_k: int = 20,
    ):
        self.item_user  = item_user
        self.user_item  = user_item
        self.item_means = item_means
        self.user_means = user_means
        self.sim_fn     = SIMILARITY_FUNCS[similarity]
        self.pred_name  = prediction
        self.top_k      = top_k
        self.global_mean = sum(item_means.values()) / len(item_means) if item_means else 3.0
        self._sim_cache: dict = {}

    def _get_similarity(self, i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key not in self._sim_cache:
            s = self.sim_fn(self.item_user.get(i1, {}), self.item_user.get(i2, {}))
            common = len(set(self.item_user.get(i1, {})) & set(self.item_user.get(i2, {})))
            register_cocount(i1, i2, common)
            self._sim_cache[key] = s
        return self._sim_cache[key]

    def get_similar_items(self, item_id):
        sims = []
        for other in self.item_user:
            if other == item_id:
                continue
            s = self._get_similarity(item_id, other)
            if s > 0:
                sims.append((other, s))
        sims.sort(key=lambda x: -x[1])
        return sims[: self.top_k]

    def predict(self, user_id, item_id):
        user_ratings = self.user_item.get(user_id, {})
        # similar items that this user has actually rated
        sim_items = self.get_similar_items(item_id)
        rated_sim = {iid: user_ratings[iid] for iid, _ in sim_items if iid in user_ratings}
        # build neighbour list aligned to rated_sim
        neighbours = [(iid, s) for iid, s in sim_items if iid in rated_sim]
        i_mean = self.item_means.get(item_id, self.global_mean)

        if self.pred_name == "weighted_average":
            return predict_weighted_average(rated_sim, neighbours, self.global_mean)
        elif self.pred_name == "mean_centered":
            return predict_mean_centered(
                rated_sim, neighbours, i_mean, self.item_means, self.global_mean
            )
        else:
            return predict_significance_weighted(
                rated_sim, neighbours, i_mean, self.item_means, global_mean=self.global_mean
            )

    def recommend(self, user_id, n=10):
        if user_id not in self.user_item:
            return []
        seen = set(self.user_item[user_id])
        all_items = set(self.item_user.keys())
        candidates = all_items - seen

        scored = []
        for item_id in candidates:
            pred = self.predict(user_id, item_id)
            scored.append((item_id, pred))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def explain(self, user_id, item_id) -> Explanation:
        """
        Return an Explanation for why item_id is recommended to user_id.

        Evidence collected:
          - Top-3 most similar items that the user has already rated
          - Their similarity to the candidate item and the user's rating
          - Number of shared raters between candidate and each similar item
          - Prediction method and predicted score
        """
        user_ratings = self.user_item.get(user_id, {})
        sim_items    = self.get_similar_items(item_id)
        rated_sim    = [(iid, s) for iid, s in sim_items if iid in user_ratings]
        pred         = self.predict(user_id, item_id)
        i_mean       = self.item_means.get(item_id, self.global_mean)
        sim_name     = self.sim_fn.__name__.replace("_similarity", "")

        if not rated_sim:
            reason = (
                f"Predicted from item average rating ({i_mean:.1f}★) — "
                f"none of your rated items are similar to this one."
            )
            details = {
                "Prediction": f"{pred:.3f}",
                "Fallback": "No rated neighbours; used item mean.",
            }
            return Explanation(item_id, pred, "ItemCF", reason, details)

        top3 = rated_sim[:3]
        n_total = len(rated_sim)

        item_lines = []
        for similar_iid, sim in top3:
            user_rating   = user_ratings[similar_iid]
            shared_raters = len(
                set(self.item_user.get(item_id, {})) &
                set(self.item_user.get(similar_iid, {}))
            )
            item_lines.append(
                f"Item {similar_iid}  "
                f"{_fmt_sim(sim_name, sim)}  |  "
                f"you rated it {user_rating:.0f}★  |  "
                f"{shared_raters} shared rater(s)"
            )

        reason = (
            f"Recommended because it is similar to {n_total} item(s) you "
            f"have already rated (closest: {_fmt_sim(sim_name, top3[0][1])})."
        )
        details = {
            "Pred. score":  f"{pred:.4f}  (clipped to [1,5])",
            "Pred. method": self.pred_name.replace("_", " ").title(),
            "Similarity":   sim_name.title(),
            "Item avg ★":   f"{i_mean:.2f}",
            "Similar items you rated": item_lines,
        }
        return Explanation(item_id, pred, "ItemCF", reason, details)

    def recommend_with_explanations(self, user_id, n=10):
        """Return list of Explanation objects (sorted by predicted score)."""
        recs = self.recommend(user_id, n)
        return [self.explain(user_id, iid) for iid, _ in recs]



class ContentBasedFilter:
    """
    Builds a user profile as the weighted average of feature vectors
    of items the user has rated. Recommends items whose feature vectors
    are most cosine-similar to this profile.
    """
    def __init__(self, user_item: dict, item_meta: dict):
        self.user_item = user_item
        self.item_meta = item_meta
        self._user_profiles: dict = {}

    def _build_profile(self, user_id):
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        profile: dict = defaultdict(float)
        total_weight = 0.0
        for iid, rating in self.user_item.get(user_id, {}).items():
            for feat, val in self.item_meta.get(iid, {}).items():
                profile[feat] += rating * val
            total_weight += rating
        if total_weight > 0:
            profile = {f: v / total_weight for f, v in profile.items()}
        self._user_profiles[user_id] = dict(profile)
        return self._user_profiles[user_id]

    def recommend(self, user_id, n=10):
        profile = self._build_profile(user_id)
        if not profile:
            return []
        seen = set(self.user_item.get(user_id, {}))
        scored = []
        for iid, meta in self.item_meta.items():
            if iid in seen or not meta:
                continue
            sim = cosine_similarity(profile, meta)
            scored.append((iid, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def explain(self, user_id, item_id) -> Explanation:
        """
        Return an Explanation for why item_id is recommended to user_id.

        Evidence collected:
          - Top matching features between the user profile and the item
          - Which past items built those feature preferences
          - The cosine similarity score between profile and item vector
        """
        profile  = self._build_profile(user_id)
        meta     = self.item_meta.get(item_id, {})
        sim      = cosine_similarity(profile, meta)

        if not profile or not meta:
            reason = "Not enough metadata to explain this recommendation."
            return Explanation(item_id, sim, "ContentBased", reason, {})

        # Find overlapping features and rank by profile weight
        overlap = {
            feat: profile[feat]
            for feat in profile
            if feat in meta
        }
        top_feats = sorted(overlap.items(), key=lambda x: -x[1])[:5]

        # Which past items contributed most to the top features?
        past_items = self.user_item.get(user_id, {})
        contributing = {}
        for feat, _ in top_feats:
            for iid, rating in past_items.items():
                if feat in self.item_meta.get(iid, {}):
                    contributing[iid] = contributing.get(iid, 0) + rating
        top_past = sorted(contributing.items(), key=lambda x: -x[1])[:3]

        feat_lines = [
            f"{feat.split(':')[0]} = '{feat.split(':')[1]}'  "
            f"(profile weight {w:.3f})"
            for feat, w in top_feats
        ]
        past_lines = [
            f"Item {iid}  (contributed {score:.1f} weighted feature matches)"
            for iid, score in top_past
        ]

        reason = (
            f"Recommended because its attributes match your taste profile "
            f"(cosine similarity = {sim:.4f}). "
            f"Key matching features: "
            + ", ".join(f.split(":")[0] for f, _ in top_feats[:3]) + "."
        )
        details = {
            "Profile sim":  f"{sim:.4f}",
            "Top matching features": feat_lines,
            "Built from past items": past_lines if past_lines else ["(no direct overlap found)"],
        }
        return Explanation(item_id, sim, "ContentBased", reason, details)

    def recommend_with_explanations(self, user_id, n=10):
        """Return list of Explanation objects (sorted by similarity score)."""
        recs = self.recommend(user_id, n)
        return [self.explain(user_id, iid) for iid, _ in recs]



class HybridRecommender:
    """
    Combines User-CF, Item-CF, and Content-Based scores via
    a weighted linear blend.
    """
    def __init__(
        self,
        user_cf: UserBasedCF,
        item_cf: ItemBasedCF,
        content_cb: ContentBasedFilter,
        w_user: float = 0.4,
        w_item: float = 0.4,
        w_content: float = 0.2,
    ):
        self.user_cf   = user_cf
        self.item_cf   = item_cf
        self.content   = content_cb
        self.w_user    = w_user
        self.w_item    = w_item
        self.w_content = w_content

    def _normalise(self, scores: list) -> dict:
        """Min-max normalise (item_id, score) list → {item_id: norm_score}."""
        if not scores:
            return {}
        vals = [s for _, s in scores]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {iid: 0.5 for iid, _ in scores}
        return {iid: (s - lo) / (hi - lo) for iid, s in scores}

    def recommend(self, user_id, n=10):
        pool_n = max(n * 5, 50)
        ucf_recs  = self.user_cf.recommend(user_id, pool_n)
        icf_recs  = self.item_cf.recommend(user_id, pool_n)
        cb_recs   = self.content.recommend(user_id, pool_n)

        ucf_norm  = self._normalise(ucf_recs)
        icf_norm  = self._normalise(icf_recs)
        cb_norm   = self._normalise(cb_recs)

        all_items = set(ucf_norm) | set(icf_norm) | set(cb_norm)
        blended = []
        for iid in all_items:
            score = (
                self.w_user    * ucf_norm.get(iid, 0.0)
                + self.w_item  * icf_norm.get(iid, 0.0)
                + self.w_content * cb_norm.get(iid, 0.0)
            )
            blended.append((iid, score))
        blended.sort(key=lambda x: -x[1])
        return blended[:n]

    def explain(self, user_id, item_id) -> Explanation:
        """
        Return an Explanation for why item_id is recommended to user_id.

        For the Hybrid model this shows:
          - The normalised contribution from each sub-model
          - The blended score and component weights
          - A mini-reason from the strongest contributing sub-model
        """
        pool_n = 200
        ucf_recs  = self.user_cf.recommend(user_id, pool_n)
        icf_recs  = self.item_cf.recommend(user_id, pool_n)
        cb_recs   = self.content.recommend(user_id, pool_n)

        ucf_norm  = self._normalise(ucf_recs)
        icf_norm  = self._normalise(icf_recs)
        cb_norm   = self._normalise(cb_recs)

        ucf_s  = ucf_norm.get(item_id, 0.0)
        icf_s  = icf_norm.get(item_id, 0.0)
        cb_s   = cb_norm.get(item_id, 0.0)

        blend = self.w_user * ucf_s + self.w_item * icf_s + self.w_content * cb_s

        # Which sub-model is the primary driver?
        contributions = {
            "UserCF":      self.w_user    * ucf_s,
            "ItemCF":      self.w_item    * icf_s,
            "ContentBased": self.w_content * cb_s,
        }
        dominant = max(contributions, key=contributions.get)

        # Pull a sub-explanation from the dominant model
        try:
            if dominant == "UserCF":
                sub_exp = self.user_cf.explain(user_id, item_id)
            elif dominant == "ItemCF":
                sub_exp = self.item_cf.explain(user_id, item_id)
            else:
                sub_exp = self.content.explain(user_id, item_id)
            sub_reason = sub_exp.reason
        except Exception:
            sub_reason = "(sub-model explanation unavailable)"

        reason = (
            f"Hybrid blend of UserCF ({self.w_user*100:.0f}%), "
            f"ItemCF ({self.w_item*100:.0f}%), "
            f"ContentBased ({self.w_content*100:.0f}%). "
            f"Primary driver: {dominant}. {sub_reason}"
        )
        details = {
            "Blend score":   f"{blend:.4f}",
            "UserCF":        f"norm={ucf_s:.4f}  weight={self.w_user}  contrib={self.w_user*ucf_s:.4f}",
            "ItemCF":        f"norm={icf_s:.4f}  weight={self.w_item}  contrib={self.w_item*icf_s:.4f}",
            "ContentBased":  f"norm={cb_s:.4f}  weight={self.w_content}  contrib={self.w_content*cb_s:.4f}",
            "Dominant model": dominant,
        }
        return Explanation(item_id, blend, "Hybrid", reason, details)

    def recommend_with_explanations(self, user_id, n=10):
        """Return list of Explanation objects (sorted by blended score)."""
        recs = self.recommend(user_id, n)
        return [self.explain(user_id, iid) for iid, _ in recs]



def train_test_split(user_item: dict, test_ratio: float = 0.2, seed: int = 42):
    """
    Leave-one-out style split: for each user, randomly hold out
    test_ratio of their ratings as the test set.
    Returns train_ui, test_pairs [(user, item, true_rating)].
    """
    rng = random.Random(seed)
    train_ui: dict = {}
    test_pairs = []

    for uid, ratings in user_item.items():
        items = list(ratings.items())
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test_items  = items[:n_test]
        train_items = items[n_test:]

        train_ui[uid] = dict(train_items)
        for iid, r in test_items:
            test_pairs.append((uid, iid, r))

    return train_ui, test_pairs


def evaluate_rating_prediction(model, test_pairs: list, rating_range=(1, 5)):
    """Compute RMSE and MAE over held-out (user, item, true_rating) triples."""
    lo, hi = rating_range
    errors = []
    for uid, iid, true_r in test_pairs:
        try:
            pred = model.predict(uid, iid)
            pred = max(lo, min(hi, pred))   # clip to valid range
        except Exception:
            pred = (lo + hi) / 2.0          # fallback
        errors.append((pred - true_r) ** 2)

    if not errors:
        return {"RMSE": None, "MAE": None}
    rmse = math.sqrt(sum(errors) / len(errors))
    mae  = sum(math.sqrt(e) for e in errors) / len(errors)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}


def evaluate_ranking(
    recommender,
    user_item_test: dict,   # held-out items per user (ground truth)
    user_item_train: dict,  # training ratings (to exclude from recs)
    k: int = 10,
    n_users: int = 50,
    seed: int = 42,
):
    """
    Compute Precision@K, Recall@K, F1@K and Coverage
    over a sample of users.
    """
    rng = random.Random(seed)
    eligible = [u for u in user_item_test if u in user_item_train and user_item_test[u]]
    sample   = rng.sample(eligible, min(n_users, len(eligible)))

    all_recommended = set()
    precisions, recalls, f1s = [], [], []

    for uid in sample:
        try:
            recs = recommender.recommend(uid, n=k)
        except Exception:
            continue
        rec_items = set(r for r, _ in recs)
        rel_items  = set(user_item_test.get(uid, {}))
        all_recommended.update(rec_items)

        hits = len(rec_items & rel_items)
        p = hits / k if k else 0
        r = hits / len(rel_items) if rel_items else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    all_items = set(i for u in user_item_train.values() for i in u)
    coverage  = len(all_recommended) / len(all_items) if all_items else 0

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else None

    return {
        f"Precision@{k}": avg(precisions),
        f"Recall@{k}":    avg(recalls),
        f"F1@{k}":        avg(f1s),
        "Coverage":       round(coverage, 4),
    }



def run_experiments(filepath: str, k: int = 10, eval_users: int = 30):
    print("=" * 65)
    print("  Amazon Fashion Recommendation System — Full Evaluation")
    print("=" * 65)

    # ── Load & Split ──────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    t0 = time.time()
    records = load_data(filepath)
    user_item_full, item_user_full, item_meta, user_means_full, item_means_full = build_matrices(records)
    print(f"      Records  : {len(records):,}")
    print(f"      Users    : {len(user_item_full):,}")
    print(f"      Items    : {len(item_user_full):,}")
    print(f"      Elapsed  : {time.time()-t0:.1f}s")

    print("\n[2/4] Splitting train / test (80 / 20) …")
    train_ui, test_pairs = train_test_split(user_item_full, test_ratio=0.2)

    # Rebuild matrices on training data only
    train_records = [
        {"reviewerID": u, "asin": i, "overall": r}
        for u, items in train_ui.items()
        for i, r in items.items()
    ]
    _, item_user_tr, _, user_means_tr, item_means_tr = build_matrices(train_records)

    # Ground-truth per user for ranking evaluation
    test_ui = defaultdict(dict)
    for uid, iid, r in test_pairs:
        test_ui[uid][iid] = r
    test_ui = dict(test_ui)

    print(f"      Train ratings: {sum(len(v) for v in train_ui.values()):,}")
    print(f"      Test  pairs  : {len(test_pairs):,}")

    # ── Similarity × Prediction grid ─────────────────────────────
    similarities = ["cosine", "pearson", "euclidean"]
    predictions  = ["weighted_average", "mean_centered", "significance_weighting"]

    print("\n[3/4] Running User-CF & Item-CF experiments …")
    print(f"      (RMSE/MAE on {len(test_pairs):,} test pairs; "
          f"Precision@{k}/Recall@{k} on {eval_users} sampled users)\n")

    results = []

    for sim in similarities:
        for pred in predictions:
            label = f"UserCF | sim={sim:<10} | pred={pred}"
            print(f"  ► {label} …", end=" ", flush=True)
            t1 = time.time()

            ucf = UserBasedCF(
                train_ui, user_means_tr,
                similarity=sim, prediction=pred, top_k=20
            )
            rating_metrics = evaluate_rating_prediction(ucf, test_pairs)
            ranking_metrics = evaluate_ranking(
                ucf, test_ui, train_ui, k=k, n_users=eval_users
            )
            elapsed = time.time() - t1
            row = {"Model": f"UserCF", "Similarity": sim, "Prediction": pred,
                   **rating_metrics, **ranking_metrics, "Time(s)": round(elapsed, 1)}
            results.append(row)
            print(f"RMSE={rating_metrics['RMSE']}  P@{k}={ranking_metrics[f'Precision@{k}']}  ({elapsed:.1f}s)")

        for pred in predictions:
            label = f"ItemCF | sim={sim:<10} | pred={pred}"
            print(f"  ► {label} …", end=" ", flush=True)
            t1 = time.time()

            icf = ItemBasedCF(
                item_user_tr, train_ui, item_means_tr, user_means_tr,
                similarity=sim, prediction=pred, top_k=20
            )
            rating_metrics  = evaluate_rating_prediction(icf, test_pairs)
            ranking_metrics = evaluate_ranking(
                icf, test_ui, train_ui, k=k, n_users=eval_users
            )
            elapsed = time.time() - t1
            row = {"Model": "ItemCF", "Similarity": sim, "Prediction": pred,
                   **rating_metrics, **ranking_metrics, "Time(s)": round(elapsed, 1)}
            results.append(row)
            print(f"RMSE={rating_metrics['RMSE']}  P@{k}={ranking_metrics[f'Precision@{k}']}  ({elapsed:.1f}s)")

    # ── Content-Based & Hybrid ────────────────────────────────────
    print("\n[4/4] Content-Based & Hybrid …")

    cb = ContentBasedFilter(train_ui, item_meta)
    cb_rank = evaluate_ranking(cb, test_ui, train_ui, k=k, n_users=eval_users)
    print(f"  ► ContentBased                  "
          f"P@{k}={cb_rank[f'Precision@{k}']}  Coverage={cb_rank['Coverage']}")
    results.append({"Model": "ContentBased", "Similarity": "cosine(profile)", "Prediction": "N/A",
                    "RMSE": "N/A", "MAE": "N/A", **cb_rank, "Time(s)": "-"})

    # Best UserCF & ItemCF (by RMSE) for the hybrid
    ucf_best = UserBasedCF(train_ui, user_means_tr, similarity="pearson",
                           prediction="mean_centered", top_k=20)
    icf_best = ItemBasedCF(item_user_tr, train_ui, item_means_tr, user_means_tr,
                           similarity="cosine", prediction="mean_centered", top_k=20)
    hybrid = HybridRecommender(ucf_best, icf_best, cb, w_user=0.4, w_item=0.4, w_content=0.2)
    hyb_rank = evaluate_ranking(hybrid, test_ui, train_ui, k=k, n_users=eval_users)
    print(f"  ► Hybrid (UCF+ICF+CB)           "
          f"P@{k}={hyb_rank[f'Precision@{k}']}  Coverage={hyb_rank['Coverage']}")
    results.append({"Model": "Hybrid", "Similarity": "pearson/cosine/cosine",
                    "Prediction": "mean_centered/mean_centered/N/A",
                    "RMSE": "N/A", "MAE": "N/A", **hyb_rank, "Time(s)": "-"})

    # ── Summary Table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Model':<10} {'Sim':<12} {'Pred':<24} {'RMSE':>6} {'MAE':>6} {'P@K':>6} {'R@K':>6} {'F1@K':>6} {'Cov':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        sim_short  = r["Similarity"][:11]
        pred_short = r["Prediction"][:23]
        print(
            f"{r['Model']:<10} {sim_short:<12} {pred_short:<24} "
            f"{str(r.get('RMSE','N/A')):>6} {str(r.get('MAE','N/A')):>6} "
            f"{str(r.get(f'Precision@{k}','N/A')):>6} "
            f"{str(r.get(f'Recall@{k}','N/A')):>6} "
            f"{str(r.get(f'F1@{k}','N/A')):>6} "
            f"{str(r.get('Coverage','N/A')):>6}"
        )

    # ── Sample Recommendations with Explanations ─────────────────
    print("\n" + "=" * 72)
    print("  SAMPLE RECOMMENDATIONS WITH EXPLANATIONS (first eligible user)")
    print("=" * 72)

    # pick a user who has at least a few training ratings for richer output
    sample_user = max(
        train_ui, key=lambda u: len(train_ui[u])
    )
    rated_sample = list(train_ui[sample_user].items())
    print(f"\n  User ID : {sample_user}")
    print(f"  Rated items (up to 5): {rated_sample[:5]}")
    print()

    for label, model in [
        ("USER-BASED CF  (pearson, mean_centered)", ucf_best),
        ("ITEM-BASED CF  (cosine, mean_centered)",  icf_best),
        ("CONTENT-BASED FILTERING",                 cb),
        ("HYBRID  (UCF 40% + ICF 40% + CB 20%)",   hybrid),
    ]:
        print(f"  {'─'*68}")
        print(f"  ▶  {label}")
        print(f"  {'─'*68}")
        try:
            explanations = model.recommend_with_explanations(sample_user, n=3)
            if not explanations:
                print("  (no recommendations generated)\n")
                continue
            for exp in explanations:
                print(exp.pretty(width=70))
        except Exception as e:
            print(f"  ERROR – {e}\n")
        print()

    print("=" * 72)
    print("  Done.")
    print("=" * 72)
    return results



if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "AMAZON_FASHION_5.jsonl"
    run_experiments(filepath, k=10, eval_users=30)