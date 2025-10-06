
import streamlit as st
import pandas as pd
import folium
from folium import CircleMarker, Marker, PolyLine, Popup, Icon
from streamlit_folium import st_folium
import math
from io import BytesIO

st.set_page_config(page_title="Mahindra NRC RO Planner (Fixed Clustering)", layout="wide")
st.title("Mahindra NRC RO Planner — Fixed clustering by Max RO")

st.sidebar.header("Inputs / Controls")
max_ro = st.sidebar.slider("Max RO per cluster", 1000, 10000, 6000, step=500)
min_distance_km = st.sidebar.slider("Min distance from existing workshop (km)", 1, 20, 5)

# Try to load local files included in the repo; if not present allow upload
try:
    df_ws = pd.read_excel("KMA_Mahindra_Workshops_Lat_Long.xlsx")
    df_proj = pd.read_excel("KMA_NRC_F30_Retail_RO_Projections_PV_Lat_Long_Pincode (1).xlsx")
    st.sidebar.success("Loaded data from local files packaged in the repo.")
except Exception:
    st.sidebar.info("Local files not found — please upload Excel files below.")
    uploaded_ws = st.sidebar.file_uploader("Upload Mahindra workshops Excel", type=["xlsx","xls"])
    uploaded_proj = st.sidebar.file_uploader("Upload NRC projection Excel", type=["xlsx","xls"])
    if uploaded_ws and uploaded_proj:
        df_ws = pd.read_excel(uploaded_ws)
        df_proj = pd.read_excel(uploaded_proj)
    else:
        st.stop()

# normalize column names
df_ws.columns = [c.strip() if isinstance(c, str) else c for c in df_ws.columns]
df_proj.columns = [c.strip() if isinstance(c, str) else c for c in df_proj.columns]

# helper to find a column by list of candidates (case-insensitive)
def find_col(df, candidates):
    cols = {col.lower(): col for col in df.columns if isinstance(col, str)}
    for cand in candidates:
        if cand in df.columns:
            return cand
    for cand in candidates:
        low = cand.lower()
        if low in cols:
            return cols[low]
    # try substring matching
    for col in df.columns:
        for cand in candidates:
            if isinstance(col, str) and cand.lower() in col.lower():
                return col
    return None

# expected columns (from your uploaded files)
ws_name_col = find_col(df_ws, ["Mabindra Workshop Location", "Mahindra Workshop Location", "workshop", "name"])
ws_pincode_col = find_col(df_ws, ["Pincode", "Pin Code", "pincode"])
ws_lat_col = find_col(df_ws, ["Latitude", "Lat"])
ws_lon_col = find_col(df_ws, ["Longitude", "Lon", "Lng"])

proj_pincode_col = find_col(df_proj, ["Customer Pin Code", "Pincode", "Pin Code", "pincode"])
proj_lat_col = find_col(df_proj, ["Latitude", "Lat"])
proj_lon_col = find_col(df_proj, ["Longitude", "Lon", "Lng"])
proj_ro_col = find_col(df_proj, ["Projected NRC RO", "Projected_NRC_RO", "Projected NRC", "F30_RO_Projection", "Proj_RO", "Projected_RO", "NRC_RO"])

missing = []
for label, col in [
    ("workshop name", ws_name_col),
    ("workshop pincode", ws_pincode_col),
    ("workshop lat", ws_lat_col),
    ("workshop lon", ws_lon_col),
    ("proj pincode", proj_pincode_col),
    ("proj lat", proj_lat_col),
    ("proj lon", proj_lon_col),
    ("proj ro", proj_ro_col),
]:
    if col is None:
        missing.append(label)

if missing:
    st.error("Could not confidently detect these columns: " + ", ".join(missing))
    st.write("Workshop file columns:", list(df_ws.columns))
    st.write("Projection file columns:", list(df_proj.columns))
    st.stop()

# rename for internal use
df_ws = df_ws.rename(columns={ws_name_col: "workshop_name", ws_pincode_col: "pincode", ws_lat_col: "lat", ws_lon_col: "lon"})
df_proj = df_proj.rename(columns={proj_pincode_col: "pincode", proj_lat_col: "lat", proj_lon_col: "lon", proj_ro_col: "projected_ro"})

# ensure numeric
df_proj["projected_ro"] = pd.to_numeric(df_proj["projected_ro"], errors="coerce").fillna(0)
df_proj["lat"] = pd.to_numeric(df_proj["lat"], errors="coerce")
df_proj["lon"] = pd.to_numeric(df_proj["lon"], errors="coerce")
df_ws["lat"] = pd.to_numeric(df_ws["lat"], errors="coerce")
df_ws["lon"] = pd.to_numeric(df_ws["lon"], errors="coerce")

df_proj = df_proj.dropna(subset=["lat","lon"]).reset_index(drop=True)
df_ws = df_ws.dropna(subset=["lat","lon"]).reset_index(drop=True)

# haversine distance
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# greedy clustering by projected_ro cap (this honors max_ro slider)
def greedy_spatial_clustering(df, max_ro=6000):
    working = df.copy().reset_index(drop=True)
    unassigned = set(working.index.tolist())
    clusters = []
    while unassigned:
        seed = max(unassigned, key=lambda idx: working.at[idx, "projected_ro"])
        members = [seed]
        unassigned.remove(seed)
        total = float(working.at[seed, "projected_ro"])
        centroid_lat = float(working.at[seed, "lat"])
        centroid_lon = float(working.at[seed, "lon"])
        while total < max_ro and unassigned:
            nearest = min(unassigned, key=lambda idx: haversine_km(centroid_lat, centroid_lon, working.at[idx, "lat"], working.at[idx, "lon"]))
            members.append(nearest)
            unassigned.remove(nearest)
            total += float(working.at[nearest, "projected_ro"])
            weights = working.loc[members, "projected_ro"].astype(float).values
            lats = working.loc[members, "lat"].astype(float).values
            lons = working.loc[members, "lon"].astype(float).values
            if weights.sum() > 0:
                centroid_lat = (lats * weights).sum() / weights.sum()
                centroid_lon = (lons * weights).sum() / weights.sum()
            else:
                centroid_lat = lats.mean()
                centroid_lon = lons.mean()
        members_df = working.loc[members].copy().reset_index(drop=True)
        clusters.append({ "members": members_df, "total_ro": float(total), "centroid": (float(centroid_lat), float(centroid_lon)) })
    return clusters

clusters = greedy_spatial_clustering(df_proj, max_ro=max_ro)

st.write(f"Formed {len(clusters)} clusters with max_ro={max_ro}")

# Determine centroid pincode and suggest locations filtered by min_distance_km
def find_centroid_pincode(members_df, centroid):
    if members_df.empty:
        return None, None, None
    distances = members_df.apply(lambda r: haversine_km(centroid[0], centroid[1], r["lat"], r["lon"]), axis=1)
    idx = distances.idxmin()
    row = members_df.loc[idx]
    return row.get("pincode"), float(row.get("lat")), float(row.get("lon"))

def nearest_distance_to_workshops(lat, lon, workshops_df):
    if workshops_df.empty:
        return float('inf')
    dists = workshops_df.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
    return float(dists.min())

suggestions = []
for i, c in enumerate(clusters, start=1):
    centroid = c["centroid"]
    centroid_pincode, centroid_lat, centroid_lon = find_centroid_pincode(c["members"], centroid)
    dist_to_nearest = nearest_distance_to_workshops(centroid_lat, centroid_lon, df_ws)
    suggestions.append({
        "cluster_id": i,
        "centroid_pincode": centroid_pincode,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "total_ro": c["total_ro"],
        "dist_to_nearest_workshop_km": dist_to_nearest,
        "suggested": dist_to_nearest >= min_distance_km
    })

sug_df = pd.DataFrame(suggestions)

# MAP
st.subheader("Interactive Map")
center_lat = float(df_proj["lat"].mean())
center_lon = float(df_proj["lon"].mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

# clusters
if st.sidebar.checkbox("Show clusters", value=True):
    for i, c in enumerate(clusters, start=1):
        members = c["members"]
        total_ro = c["total_ro"]
        centroid = c["centroid"]
        popup_html = f"Cluster {i}<br/>Total RO: {int(total_ro)}<br/>Members: {len(members)}"
        CircleMarker(location=[centroid[0], centroid[1]], radius=8, fill=True, fill_opacity=0.9, popup=Popup(popup_html)).add_to(m)
        for _, row in members.iterrows():
            PolyLine(locations=[[centroid[0], centroid[1]],[row["lat"], row["lon"]]], weight=1).add_to(m)

# existing workshops
if st.sidebar.checkbox("Show existing workshops", value=True):
    for _, r in df_ws.iterrows():
        tooltip = f"{r.get('workshop_name','')} (Pincode: {r.get('pincode','')})"
        Marker(location=[r["lat"], r["lon"]], tooltip=tooltip, icon=Icon(color="red",icon="wrench", prefix='fa')).add_to(m)

# suggested
if st.sidebar.checkbox("Show suggested locations", value=True):
    for _, s in sug_df[sug_df["suggested"]].iterrows():
        popup = f"Cluster {s['cluster_id']}<br/>RO: {int(s['total_ro'])}<br/>Pincode: {s['centroid_pincode']}<br/>Dist to nearest WS: {s['dist_to_nearest_workshop_km']:.2f} km"
        Marker(location=[s["centroid_lat"], s["centroid_lon"]], tooltip=popup, icon=Icon(color="green", icon="plus", prefix='fa')).add_to(m)

st_folium(m, width=1000, height=600)

# Export
st.subheader("Export suggested locations and cluster detail")
csv = sug_df.to_csv(index=False).encode('utf-8')
st.download_button("Download suggested locations CSV", data=csv, file_name="suggested_locations.csv", mime="text/csv")

all_clusters = []
for i, c in enumerate(clusters, start=1):
    members = c["members"].copy()
    members["cluster_id"] = i
    members["cluster_total_ro"] = c["total_ro"]
    all_clusters.append(members)
all_clusters_df = pd.concat(all_clusters, ignore_index=True)
st.download_button("Download clusters detail CSV", data=all_clusters_df.to_csv(index=False).encode('utf-8'), file_name="clusters_detail.csv", mime="text/csv")
