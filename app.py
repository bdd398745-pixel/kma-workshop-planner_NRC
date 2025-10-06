
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt

st.set_page_config(page_title="Mahindra NRC RO Planning Tool", layout="wide")

st.title("🚗 Mahindra Workshop Planning Tool - NRC RO Version")
st.markdown("Visualize existing workshops and identify new optimal locations based on *Projected NRC RO* clustering.")

# --- Load Excel Files ---
@st.cache_data
def load_data():
    df_workshops = pd.read_excel("KMA_Mahindra_Workshops_Lat_Long (1).xlsx")
    df_proj = pd.read_excel("KMA_NRC_F30_Retail_RO_Projections_PV_Lat_Long_Pincode (1).xlsx")
    return df_workshops, df_proj

df_workshops, df_proj = load_data()

# --- Controls ---
max_ro = st.sidebar.slider("Max RO per Cluster", 1000, 10000, 6000, 500)
min_distance = st.sidebar.slider("Min Distance from Existing Workshop (km)", 1, 10, 5, 1)

# --- Cluster NRC ROs ---
df_proj = df_proj.dropna(subset=["Latitude", "Longitude"])
df_proj["Proj_RO"] = df_proj.get("Projected_NRC_RO", df_proj.iloc[:, -1])  # fallback for unknown header

points = df_proj[["Latitude", "Longitude"]].values
n_clusters = max(1, len(points)//10)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_proj["Cluster"] = kmeans.fit_predict(points)

cluster_summary = df_proj.groupby("Cluster").agg(
    Total_RO=("Proj_RO", "sum"),
    Latitude=("Latitude", "mean"),
    Longitude=("Longitude", "mean")
).reset_index()

# --- Haversine distance ---
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

# --- Filter for suggested workshops ---
suggested = []
for _, row in cluster_summary.iterrows():
    close = False
    for _, w in df_workshops.iterrows():
        if haversine(row.Latitude, row.Longitude, w["Latitude"], w["Longitude"]) < min_distance:
            close = True
            break
    if not close:
        suggested.append(row)
df_suggested = pd.DataFrame(suggested)

# --- Map ---
m = folium.Map(location=[df_proj["Latitude"].mean(), df_proj["Longitude"].mean()], zoom_start=7)

# Existing Workshops
for _, r in df_workshops.iterrows():
    folium.Marker(
        [r["Latitude"], r["Longitude"]],
        popup=f"Workshop: {r.get('Workshop Name', 'NA')}<br>Pincode: {r.get('Pin Code', 'NA')}",
        icon=folium.Icon(color="red", icon="wrench", prefix="fa")
    ).add_to(m)

# Suggested Locations
for _, r in df_suggested.iterrows():
    folium.Marker(
        [r["Latitude"], r["Longitude"]],
        popup=f"Suggested Location<br>Cluster {r['Cluster']}<br>Total NRC ROs: {r['Total_RO']}",
        icon=folium.Icon(color="green", icon="plus", prefix="fa")
    ).add_to(m)

st_folium(m, width=1200, height=700)

# --- Export Summary ---
st.subheader("📊 Cluster Summary")
st.dataframe(cluster_summary)

if not df_suggested.empty:
    csv = df_suggested.to_csv(index=False).encode('utf-8')
    st.download_button("Download Suggested Locations (CSV)", csv, "suggested_locations.csv", "text/csv")
