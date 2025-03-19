import streamlit as st
import osmnx as ox
import folium
import pandas as pd
import os
import networkx as nx
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from folium import LayerControl
import requests
import googlemaps
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Google Maps API Client
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
# Initialize session state
for key in ["start_lat", "start_lon", "end_lat", "end_lon"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Popular places
POPULAR_PLACES = ["Delhi, India", "Mumbai, India", "Bangalore, India", "Chennai, India", "Hyderabad, India"]

# Load Data
crime_data = pd.read_csv("crime_data.csv")

# ✅ Fix column names for CCTV data
if os.path.exists("cctv_locations.csv"):
    cctv_data = pd.read_csv("cctv_locations.csv")
    cctv_data.rename(columns={"Longitude": "longitude", "Latitude": "latitude"}, inplace=True)
else:
    cctv_data = pd.DataFrame(columns=["latitude", "longitude"])

police_data = pd.read_csv("bengaluru_police_stations_geocoded.csv")
feedback_data = pd.read_csv("user_feedback.csv") if os.path.exists("user_feedback.csv") else pd.DataFrame(columns=["latitude", "longitude", "feedback"])

def calculate_safety_index(crime_data):
    """Calculate safety index based on crime severity."""
    safety_index = {}
    for _, row in crime_data.iterrows():
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            severity = pd.to_numeric(str(row["severity"]).split()[0], errors="coerce") or 0
            safety_index[(lat, lon)] = safety_index.get((lat, lon), 0) + severity
        except (ValueError, TypeError, KeyError):
            continue
    return safety_index

def geocode_location(location):
    """Get latitude & longitude using Google Maps API."""
    try:
        geocode_result = gmaps.geocode(location)
        if geocode_result:
            return geocode_result[0]["geometry"]["location"]["lat"], geocode_result[0]["geometry"]["location"]["lng"]
    except Exception:
        return None, None

def get_live_location():
    """Fetch live location using IP-based geolocation."""
    try:
        response = requests.requests.post(
            f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_MAPS_API_KEY}",
            json={}
            ).json()
        return map(float, response["loc"].split(","))
    except:
        return None, None

def get_route(graph, crime_data, cctv_data, start, end, route_type="fastest"):
    """Find the safest or fastest route by adjusting street weights."""
    
    # Convert crime data to a dictionary for quick lookup
    crime_index = {(row['latitude'], row['longitude']): row['severity'] for _, row in crime_data.iterrows() if "latitude" in row and "longitude" in row}
    
    # Convert CCTV data to a set for quick lookup
    cctv_locations = set(zip(cctv_data['longitude'], cctv_data['latitude']))

    # Assign weights to graph edges
    for u, v, data in graph.edges(data=True):
        lat, lon = graph.nodes[u]['y'], graph.nodes[u]['x']
        crime_score = crime_index.get((lat, lon), 0)  # Get crime score, default to 0
        cctv_bonus = -5 if (lat, lon) in cctv_locations else 0  # Reduce weight for CCTV presence
        
        if route_type == "safest":
            data['weight'] = data['length'] * (1 + crime_score / 10) + cctv_bonus
        else:
            data['weight'] = data['length']  # Fastest route (regular distance)
    
    try:
        orig_node = ox.distance.nearest_nodes(graph, start[1], start[0])
        dest_node = ox.distance.nearest_nodes(graph, end[1], end[0])
        route = nx.shortest_path(graph, orig_node, dest_node, weight="weight")
        route_length = sum(nx.get_edge_attributes(graph, "length")[edge] for edge in zip(route[:-1], route[1:]))
        estimated_time = route_length / (50 * 1000 / 60)  # Assuming 50 km/h speed
        return [(graph.nodes[node]["y"], graph.nodes[node]["x"]) for node in route], route_length, estimated_time
    except Exception:
        return [], 0, 0

def get_street_map(start, end, crime_data, cctv_data, police_data, route_type):
    """Generate an interactive map with safest & fastest routes, crime heatmap, and police/CCTV locations."""
    
    # ✅ Increase search distance to avoid missing road data
    graph = ox.graph_from_point(start, dist=10000, network_type="walk")  

    # Calculate safety index
    safety_index = calculate_safety_index(crime_data)

    # Initialize Map
    m = folium.Map(location=start, zoom_start=14)

    # Add Heatmap
    if safety_index:
        HeatMap([[lat, lon, score] for (lat, lon), score in safety_index.items()]).add_to(m)

    # Get Routes
    safest_route, safest_length, safest_time = get_route(graph, crime_data, cctv_data, start, end, "safest")
    fastest_route, fastest_length, fastest_time = get_route(graph, crime_data, cctv_data, start, end, "fastest")

    # ✅ Debugging: Print route info
    print("Safest Route:", safest_route)
    print("Fastest Route:", fastest_route)

    # Add Start & Destination markers
    folium.Marker(start, icon=folium.Icon(color="red", icon="play"), popup="Start Location").add_to(m)
    folium.Marker(end, icon=folium.Icon(color="darkblue", icon="flag"), popup="Destination").add_to(m)

    # ✅ Ensure the safest route is drawn
    if safest_route:
        folium.PolyLine(safest_route, color="green", weight=6, opacity=0.7, 
                        popup=f"🛡️ Safest Route\nDistance: {safest_length:.2f} m\nTime: {safest_time:.2f} min").add_to(m)
        m.fit_bounds(safest_route)  # Force the map to focus on the route

    # ✅ Ensure the fastest route is drawn
    if fastest_route:
        folium.PolyLine(fastest_route, color="blue", weight=6, opacity=0.7, 
                        popup=f"🚀 Fastest Route\nDistance: {fastest_length:.2f} m\nTime: {fastest_time:.2f} min").add_to(m)
        m.fit_bounds(fastest_route)  # Ensure visibility

    # ✅ If no route was found, draw a straight line
    if not safest_route and not fastest_route:
        folium.PolyLine([start, end], color="gray", weight=4, opacity=0.6, dash_array="5,5",
                        popup="⚠️ No route found, showing direct path").add_to(m)
        m.fit_bounds([start, end])  # Zoom in on the path

    # Add CCTV & Police markers
    
    for _, row in police_data.iterrows():
        folium.Marker([row["Latitude"], row["Longitude"]], icon=folium.Icon(color="blue", icon="shield"), popup=row["Police Station"]).add_to(m)

    LayerControl().add_to(m)
    return m

### **🔹 Streamlit UI Components**
st.title("ML-Based Street Safety Index")

# Select City
location = st.selectbox("Select a city:", POPULAR_PLACES)

st.header("Find the Best Route")
use_live_location = st.checkbox("Use Live Location for Start Point")

if use_live_location:
   start_lat, start_lon = 12.836559196847052, 77.65709306836929
   st.session_state["start_lat"], st.session_state["start_lon"] = start_lat, start_lon
   st.success(f"Live location set: {start_lat}, {start_lon}")
else:
    start_location = st.text_input("Enter Start Location")

destination_location = st.text_input("Enter Destination Location")

route_type = st.radio("Choose Route Type:", ["Fastest", "Safest"])  # 🔴 Added Toggle

if st.button("Geocode Locations"):
    if not use_live_location:
        st.session_state["start_lat"], st.session_state["start_lon"] = geocode_location(start_location)

    st.session_state["end_lat"], st.session_state["end_lon"] = geocode_location(destination_location)

    if None in (st.session_state["start_lat"], st.session_state["start_lon"], st.session_state["end_lat"], st.session_state["end_lon"]):
        st.error("Invalid locations. Please check your inputs.")
    else:
        st.success("Locations successfully geocoded!")

if st.button("Show Route"):
    map_object = get_street_map(
        (st.session_state["start_lat"], st.session_state["start_lon"]),
        (st.session_state["end_lat"], st.session_state["end_lon"]),
        crime_data, cctv_data, police_data, route_type.lower()
    )
    folium_static(map_object)  
