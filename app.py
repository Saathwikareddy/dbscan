import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Pickup Location Clustering",
    layout="wide"
)

st.title("🟣 Pickup Location Clustering using DBSCAN")
st.write(
    "This application uses DBSCAN to discover dense pickup regions "
    "from New York City taxi trip data."
)

# ---------------------------------
# Load dataset (from repo)
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "NewYorkCityTaxiTripDuration.csv",
        encoding="latin1"
    )

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------------------------
# Feature selection & cleaning
# ---------------------------------
st.subheader("🧹 Data Cleaning")

required_cols = ["pickup_latitude", "pickup_longitude"]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_clean = df.dropna(subset=required_cols)

st.write(f"Total records after removing missing values: **{len(df_clean)}**")

X = df_clean[required_cols]

# ---------------------------------
# Scaling
# ---------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------
# Sidebar controls
# ---------------------------------
st.sidebar.header("⚙️ DBSCAN Settings")

eps = st.sidebar.selectbox(
    "Epsilon (eps)",
    [0.2, 0.3, 0.5]
)

min_samples = st.sidebar.slider(
    "Minimum Samples",
    min_value=3,
    max_value=10,
    value=5
)

# ---------------------------------
# DBSCAN clustering
# ---------------------------------
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

df_clean["cluster"] = labels

# ---------------------------------
# Cluster evaluation
# ---------------------------------
st.subheader("📊 Cluster Evaluation")

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_points = np.sum(labels == -1)
noise_ratio = noise_points / len(labels)

st.write(f"**Number of clusters (excluding noise):** {num_clusters}")
st.write(f"**Noise points:** {noise_points}")
st.write(f"**Noise ratio:** {noise_ratio:.2f}")

# ---------------------------------
# Silhouette score
# ---------------------------------
st.subheader("📈 Silhouette Score")

mask = labels != -1

if len(set(labels[mask])) > 1:
    score = silhouette_score(X_scaled[mask], labels[mask])
    st.success(f"Silhouette Score: {score:.3f}")
else:
    st.warning("Silhouette Score: Not Applicable")

# ---------------------------------
# Visualization
# ---------------------------------
st.subheader("🗺️ Cluster Visualization")

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    df_clean["pickup_longitude"],
    df_clean["pickup_latitude"],
    c=labels,
    cmap="tab10",
    s=12
)

ax.scatter(
    df_clean.loc[labels == -1, "pickup_longitude"],
    df_clean.loc[labels == -1, "pickup_latitude"],
    color="black",
    s=12,
    label="Noise"
)

ax.set_xlabel("Pickup Longitude")
ax.set_ylabel("Pickup Latitude")
ax.set_title(f"DBSCAN Clustering (eps={eps})")
ax.legend()

st.pyplot(fig)

# ---------------------------------
# Business interpretation
# ---------------------------------
st.subheader("🧠 Business Interpretation")

st.info(
    "Each cluster represents a dense pickup zone such as busy streets or hotspots. "
    "Noise points indicate isolated or rare pickup locations. "
    "These insights help improve demand forecasting, routing, and urban planning."
)
