"""Generate a messy synthetic shopping dataset (shopping_data.csv)."""
import pandas as pd
import numpy as np

rng = np.random.default_rng(1)
categories = ["Books", "Toys", "Fashion", "Electronics", "Sports"]

# 200 products, each with a category and a price
products = pd.DataFrame({
    "product_id": [f"P{i}" for i in range(200)],
    "category": rng.choice(categories, 200),
    "price": rng.integers(5, 200, 200),
})

rows = []
for c in range(300):
    customer_id = f"C{c:03d}"
    fav = rng.choice(categories)        # this customer's favourite category
    budget = rng.integers(20, 120)      # what they usually spend
    prev = rng.integers(0, 10)          # their previous purchases

    for _, p in products.sample(30, random_state=int(rng.integers(1_000_000))).iterrows():
        likes = 3 if p["category"] == fav else 1
        affordable = 1 if p["price"] <= budget * 1.5 else 0.3
        views = rng.poisson(likes * affordable)
        cart = rng.binomial(views, 0.4)
        bought = rng.binomial(cart, 0.7)
        rows.append([customer_id, p["product_id"], p["category"], p["price"],
                     views, cart, bought, prev])

df = pd.DataFrame(rows, columns=["customer_id", "product_id", "category", "price",
                                 "views", "cart_adds", "purchases", "previous_purchases"])
df["price"] = df["price"].astype(float)

# make the data dirty on purpose
df = pd.concat([df, df.sample(100, random_state=2)], ignore_index=True)   # duplicates
df.loc[df.sample(50, random_state=3).index, "price"] = None               # missing prices
df.loc[df.sample(10, random_state=4).index, "price"] = -1                 # invalid prices

df.to_csv("shopping_data.csv", index=False)
print("Saved", len(df), "rows")
print("Purchase rate:", round((df["purchases"] > 0).mean() * 100, 1), "%")
