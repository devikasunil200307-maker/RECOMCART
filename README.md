# 🛒 Smart Product Recommendation System

A machine learning system that predicts how likely a customer is to buy each product, ranks the
products by that probability, and shows the **Top 5 recommendations**. It works for existing
customers (by customer ID) and for brand-new customers (by describing their tastes).

| | Link |
|---|---|
| CODE | https://github.com/devikasunil200307-maker/RECOMCART |
| APP LINK| https://recomcart-shopping.streamlit.app/|

## Problem

An online shop has historical customer-product interaction data: customer ID, product ID,
category, price, number of views, cart adds, purchases, and the customer's previous purchases.
The goal is to recommend products a customer is likely to buy.

## What the system does

1. **Cleans** the data: removes duplicates, fixes invalid prices (zero or negative), fills missing values.
2. **Engineers features** that describe customer behaviour and price/category fit.
3. **Trains** a Random Forest classifier to predict `bought` (purchases > 0).
4. **Evaluates** it on a held-out 20% test set.
5. **Recommends**: scores every product the customer hasn't bought, ranks by probability, shows Top 5.

## Project structure

```
RECOMCART/
├── README.md
├── app.py               # Streamlit web app (deployment)
├── make_data.py         # generates the messy synthetic dataset
├── requirements.txt     # Python dependencie
├── shopping.py          # main pipeline: clean -> features -> train -> evaluate -> recommend (+ CLI)
└── shopping_data.csv    # dataset (9,100 rows, intentionally dirty)
```

## Data

The dataset is **synthetic** and created by `make_data.py`: 300 customers, 200 products, 5 categories.
Each customer has a hidden favourite category and budget that influence what they view, cart and buy.
Dirt is injected on purpose: ~100 duplicate rows, 50 missing prices, 10 invalid (negative) prices.
Columns: `customer_id, product_id, category, price, views, cart_adds, purchases, previous_purchases`.

## Method

**Cleaning**
- Drop exact duplicate rows
- Treat prices ≤ 0 as missing
- Fill a missing price from the same product's price in other rows, then the overall median as fallback

**Features** (the model never sees the target)

| Feature | Meaning |
|---|---|
| `price`, `price_ratio` | Product price, and price relative to the customer's usual spend |
| `previous_purchases` | Customer's past purchase count |
| `cust_total_views`, `cust_total_cart` | How active the customer is overall |
| `cust_avg_price` | Customer's typical price level |
| `cust_cat_share` | Share of the customer's views in this product's category |
| `category` (one-hot) | Product category |


**New customers** are described by favourite categories, typical budget and previous purchases;
unknown activity levels are set to the median customer.




Example (command line, new customer): choose `2`, categories `Sports,Toys`, budget `30`, previous purchases `2`.


