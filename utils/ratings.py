import os
import pandas as pd
import streamlit as st

def ratings_function(variant):
    # CSV file path
    RATINGS_LOG = "logs/ratings_log.csv"

    # Ensure CSV file exists
    if not os.path.exists(RATINGS_LOG):
        pd.DataFrame(columns=["Variant", "Rating", "Comment"]).to_csv(
            RATINGS_LOG, index=False
        )

    st.title("Rate This Prediction")

    # User rating input
    rating = st.slider("Rate the prediction:", min_value=1, max_value=5, step=1)
    comment = st.text_area("Optional comment:")

    if st.button("Submit Rating"):
        # Append rating to CSV
        new_entry = pd.DataFrame(
            [{"Variant": variant, "Rating": rating, "Comment": comment}]
        )
        new_entry.to_csv(RATINGS_LOG, mode="a", header=False, index=False)
        st.success("Thank you for your feedback! 🎉")