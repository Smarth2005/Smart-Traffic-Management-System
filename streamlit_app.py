import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from streamlit_autorefresh import st_autorefresh


# --- CONFIGURATION ---
DATA_FOLDER = Path.cwd()
CSV_PATTERN = "traffic_data_"
IMAGE_PATH = "latest_frame.jpg"
REFRESH_INTERVAL = 1  # seconds

# --- SETUP PAGE ---
st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
# Auto-refresh every 3 seconds
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")

st.title("🚦 Real-Time Traffic Monitoring Dashboard")

# --- FUNCTIONS ---
def get_latest_csv():
    files = sorted(DATA_FOLDER.glob(f"{CSV_PATTERN}*.csv"), reverse=True)
    return files[0] if files else None

def load_data():
    csv_path = get_latest_csv()
    if not csv_path or not csv_path.exists():
        st.warning("⚠️ No traffic CSV data found.")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

def show_kpis(latest_row):
    col1, col2, col3 = st.columns(3)

    col1.metric("🚗 Avg Speed", f"{latest_row['average_speed']:.1f} km/h")
    col1.metric("⚠️ Congestion", f"{latest_row['congestion_level']:.1f} %")

    col2.metric("💡 Flow Efficiency", f"{latest_row['traffic_flow_efficiency']:.1f}")
    col2.metric("😊 Satisfaction", f"{latest_row['user_satisfaction_score']:.1f}/100")

    col3.metric("⛽ Fuel Index", f"{latest_row['fuel_consumption_index']:.2f}")
    col3.metric("🌫️ CO₂ Index", f"{latest_row['co2_emissions_index']:.2f}")


# --- AUTO REFRESH ---
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
remaining = REFRESH_INTERVAL - elapsed

# --- LOAD DATA + RENDER EVERYTHING FIRST ---
data = load_data()
if not data.empty:
    latest = data.iloc[-1]
    show_kpis(latest)

    with st.expander("📊 Historical Trends"):
        fig, ax = plt.subplots(figsize=(10, 4))
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        ax.plot(data["timestamp"], data["weighted_total"], label="Weighted Flow", color="orange")
        ax.plot(data["timestamp"], data["average_speed"], label="Avg Speed", color="green")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

    with st.expander("📁 View Recent Records"):
        st.dataframe(data.tail(20))

# --- LIVE FRAME ---
st.sidebar.header("Live Camera View")
if Path(IMAGE_PATH).exists():
    st.sidebar.image(IMAGE_PATH, caption="Latest Frame", use_container_width=True)
else:
    st.sidebar.info("🕓 Waiting for latest_frame.jpg...")

# --- AUTO REFRESH TIMER AT THE END ---
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
remaining = REFRESH_INTERVAL - elapsed

with st.sidebar:
    st.markdown("⏱️ Auto-refresh in:")
    st.progress(min(1, max(0, 1 - remaining / REFRESH_INTERVAL)))
    st.write(f"{int(max(0, remaining))} seconds...")

if elapsed >= REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.rerun()

