from recommender import *

records = load_data("AMAZON_FASHION_5.jsonl")
user_item, item_user, item_meta, user_means, item_means = build_matrices(records)

ucf = UserBasedCF(user_item, user_means, similarity="pearson", prediction="mean_centered")

for exp in ucf.recommend_with_explanations("A3BN0MRGRDKM0J", n=5):
    print(exp.pretty())

exp = ucf.explain("A3BN0MRGRDKM0J", "B016XAJLVO")
print(exp.reason)
print(exp.details)