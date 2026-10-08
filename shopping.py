"""Smart Product Recommendation System: clean, engineer features, train, evaluate, recommend."""
from pathlib import Path
import pandas as pd

df = pd.read_csv(Path(__file__).parent / "shopping_data.csv")
print("Rows before cleaning:", len(df))

# ---------- CLEANING ----------
df = df.drop_duplicates()
df.loc[df["price"] <= 0, "price"] = None

# fill a missing price using the SAME product's price in other rows
df["price"] = df["price"].fillna(df.groupby("product_id")["price"].transform("median"))
# if a product has no price anywhere, fall back to the overall median
df["price"] = df["price"].fillna(df["price"].median())

print("Rows after cleaning:", len(df))
print("Missing prices left:", df["price"].isna().sum())

# ---------- FEATURES ----------
df["bought"] = (df["purchases"] > 0).astype(int)

df["cust_total_views"] = df.groupby("customer_id")["views"].transform("sum")
df["cust_total_cart"] = df.groupby("customer_id")["cart_adds"].transform("sum")
df["cust_avg_price"] = df.groupby("customer_id")["price"].transform("mean")
df["price_ratio"] = df["price"] / df["cust_avg_price"]

# how much of this customer's attention goes to THIS product's category?
cat_views = df.groupby(["customer_id", "category"])["views"].transform("sum")
df["cust_cat_share"] = cat_views / df["cust_total_views"].replace(0, 1)

# ---------- TRAIN ----------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

feature_cols = ["price", "previous_purchases", "cust_total_views", "cust_total_cart",
                "cust_avg_price", "price_ratio", "cust_cat_share"]
X = pd.concat([df[feature_cols], pd.get_dummies(df["category"])], axis=1)
y = df["bought"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Training rows:", len(X_train), "| Test rows:", len(X_test))
print("Model trained!")
print("Accuracy on test data:", round(model.score(X_test, y_test), 3))

# ---------- EVALUATE ----------
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score)

print("\nIf we always said 'not bought', accuracy would be:", round(1 - y_test.mean(), 3))

proba = model.predict_proba(X_test)[:, 1]      # probability of buying
pred = (proba >= 0.3).astype(int)              # say "will buy" if probability >= 30%

print("Confusion matrix (rows = real, columns = predicted):")
print(confusion_matrix(y_test, pred))
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("F1 score :", round(f1_score(y_test, pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, proba), 3))

# ---------- RECOMMEND ----------
def recommend(customer_id, top_n=5):
    cust_rows = df[df["customer_id"] == customer_id]
    if cust_rows.empty:
        print("Unknown customer:", customer_id)
        return None

    # all products, except ones this customer already bought
    products = df[["product_id", "category", "price"]].drop_duplicates("product_id")
    bought_ids = cust_rows.loc[cust_rows["purchases"] > 0, "product_id"]
    products = products[~products["product_id"].isin(bought_ids)].copy()

    # fill in this customer's features for every product
    c = cust_rows.iloc[0]
    products["previous_purchases"] = c["previous_purchases"]
    products["cust_total_views"] = c["cust_total_views"]
    products["cust_total_cart"] = c["cust_total_cart"]
    products["cust_avg_price"] = c["cust_avg_price"]
    products["price_ratio"] = products["price"] / c["cust_avg_price"]
    share = cust_rows.groupby("category")["views"].sum() / max(c["cust_total_views"], 1)
    products["cust_cat_share"] = products["category"].map(share).fillna(0)

    # same column layout the model was trained on
    X_new = pd.concat([products[feature_cols], pd.get_dummies(products["category"])], axis=1)
    X_new = X_new.reindex(columns=X.columns, fill_value=0)

    products["probability"] = model.predict_proba(X_new)[:, 1]
    top = products.sort_values("probability", ascending=False).head(top_n)

    print("\nTop", top_n, "recommendations for", customer_id)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        print(rank, row["product_id"], row["category"], "price", row["price"],
              "->", round(row["probability"] * 100, 1), "% chance")
    return top


def recommend_new(fav_categories, budget, previous_purchases=0, top_n=5):
    products = df[["product_id", "category", "price"]].drop_duplicates("product_id").copy()

    # a new customer has no history, so use the typical customer's activity level
    typical = df.groupby("customer_id")[["cust_total_views", "cust_total_cart"]].first().median()

    products["previous_purchases"] = previous_purchases
    products["cust_total_views"] = typical["cust_total_views"]
    products["cust_total_cart"] = typical["cust_total_cart"]
    products["cust_avg_price"] = budget
    products["price_ratio"] = products["price"] / budget
    products["cust_cat_share"] = products["category"].apply(
        lambda cat: 1 / len(fav_categories) if cat in fav_categories else 0)

    X_new = pd.concat([products[feature_cols], pd.get_dummies(products["category"])], axis=1)
    X_new = X_new.reindex(columns=X.columns, fill_value=0)

    products["probability"] = model.predict_proba(X_new)[:, 1]
    top = products.sort_values("probability", ascending=False).head(top_n)

    print("\nTop", top_n, "recommendations for a NEW customer who likes", fav_categories)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        print(rank, row["product_id"], row["category"], "price", row["price"],
              "->", round(row["probability"] * 100, 1), "% chance")
    return top


# ---------- MAIN PROGRAM ----------
if __name__ == "__main__":
    print("\n=== Smart Product Recommender ===")
    choice = input("Existing customer (1) or new customer (2)? ").strip()

    if choice == "1":
        recommend(input("Customer ID (for example C007): ").strip())
    else:
        cats = input("Favourite categories, comma separated (Books,Toys,Fashion,Electronics,Sports): ")
        fav = [c.strip() for c in cats.split(",")]
        budget = float(input("Typical price you spend per item: "))
        prev = int(input("How many things have you bought before? "))
        recommend_new(fav, budget, prev)
