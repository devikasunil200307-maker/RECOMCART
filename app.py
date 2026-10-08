"""Streamlit web app for the Smart Product Recommendation System."""
import streamlit as st
import shopping  # trains the model once when first imported

st.set_page_config(page_title="Smart Product Recommender", page_icon="🛒")
st.title("🛒 Smart Product Recommender")
st.caption("A machine learning model predicts how likely a customer is to buy each product, "
           "then shows the Top 5.")

CATEGORIES = ["Books", "Toys", "Fashion", "Electronics", "Sports"]


def show(top):
    out = top[["product_id", "category", "price", "probability"]].copy()
    out["probability"] = (out["probability"] * 100).round(1).astype(str) + " %"
    out.columns = ["Product", "Category", "Price", "Purchase probability"]
    out.index = range(1, len(out) + 1)
    st.table(out)


tab1, tab2 = st.tabs(["Existing customer", "New customer"])

with tab1:
    ids = sorted(shopping.df["customer_id"].unique())
    cid = st.selectbox("Customer ID", ids)
    if st.button("Recommend", key="existing"):
        show(shopping.recommend(cid))

with tab2:
    favs = st.multiselect("Favourite categories", CATEGORIES, default=["Books"])
    budget = st.slider("Typical price per item", 5, 200, 40)
    prev = st.number_input("Previous purchases", 0, 100, 0)
    if st.button("Recommend", key="new"):
        if favs:
            show(shopping.recommend_new(favs, float(budget), int(prev)))
        else:
            st.warning("Pick at least one category.")

with st.expander("Model quality"):
    st.write("Hold-out test results are in the README and REPORT.md of the repository.")
