import streamlit as st
import pandas as pd 

st.title("Wildfire Spread Prediction System")

@st.cache_resource
def load_model():
    model = None
    model.load_state_dict(torch.load("models/...pt", map_location="cpu")["model"])
    model.eval()
    return model


model = load_model()

option = st.radio('Select Input Mode:', ['Mark on Map', 'Upload CSV'])

if option.lower() == 'mark on map':
    st.write("Click on the map where fire starts:")
    coords = st.map()

elif option.lower() == 'upload csv':
    uploaded_file = st.file_uploader("Upload CSV with columns: lat, lon, time", type=['csv'])
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Coordinates")
    st.dataframe(df.head())
