import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LocNet: GPS Position Optimizer",
    page_icon="🛰️",
    layout="wide"
)

st.title("LocNet: GNSS Position Optimizer")
st.markdown("""
This system estimates the **True Position** using four distinct mathematical and AI architectures:
1.  **Statistical Median:** Robust baseline for static data.
2.  **Kalman Filter:** Recursive mathematical state estimation.
3.  **LSTM (RNN):** Time-series Deep Learning method that learns sequential dependencies from raw numbers.
4.  **CNN:** Converts the trajectory path into a **Time-Encoded Image** and uses a Convolutional Network to predict the position.
""")

# --- SIDEBAR & CONFIG ---
st.sidebar.header("Hyperparameters")
LOOK_BACK = st.sidebar.slider("Look-Back Window (Steps)", 5, 60, 20, help="Points used to generate the image/sequence")
IMG_SIZE = 32 # Resolution for CNN
EPOCHS = st.sidebar.slider("Training Epochs", 10, 150, 40)
LEARNING_RATE = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
HDOP_THRESHOLD = st.sidebar.slider("HDOP Filter Threshold", 1.0, 10.0, 5.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- HELPER FUNCTIONS ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',')
    return df

def find_coordinate_columns(df):
    cols = df.columns
    lat_col, lon_col, hdop_col = None, None, None
    def is_match(col_name, candidates):
        return any(cand in col_name.lower() for cand in candidates)

    for col in cols:
        clean_col = col.lower().strip()
        if not lat_col and is_match(clean_col, ['lat', 'latitude']):
            lat_col = col
        if not lon_col and is_match(clean_col, ['lon', 'lng', 'long', 'longitude']):
            lon_col = col
        if not hdop_col and is_match(clean_col, ['hdop', 'dilution']):
            hdop_col = col
    return lat_col, lon_col, hdop_col

def preprocess_data(df, lat_col, lon_col, hdop_col):
    df = df.rename(columns={lat_col: 'Lat', lon_col: 'Lon'})
    if hdop_col:
        df = df.rename(columns={hdop_col: 'HDOP'})
    
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Lon'] = pd.to_numeric(df['Lon'], errors='coerce')
    df = df.dropna(subset=['Lat', 'Lon'])
    
    # 1. HDOP Filter
    hdop_removed = 0
    if 'HDOP' in df.columns:
        df['HDOP'] = pd.to_numeric(df['HDOP'], errors='coerce')
        mask = (df['HDOP'].isna()) | (df['HDOP'] < HDOP_THRESHOLD)
        df_filtered = df[mask]
        if len(df_filtered) > 5:
            hdop_removed = len(df) - len(df_filtered)
            df = df_filtered

    # 2. Outlier Removal (Z-Score)
    outliers_removed = 0
    if len(df) > 10 and df['Lat'].std() > 0:
        lat_mean, lat_std = df['Lat'].mean(), df['Lat'].std()
        lon_mean, lon_std = df['Lon'].mean(), df['Lon'].std()
        df_clean = df[
            (np.abs(df['Lat'] - lat_mean) <= 3 * lat_std) & 
            (np.abs(df['Lon'] - lon_mean) <= 3 * lon_std)
        ].copy()
        if len(df_clean) > 5:
            outliers_removed = len(df) - len(df_clean)
            df = df_clean

    # 3. Smoothing (for Deep Learning inputs)
    window_smooth = max(1, min(5, len(df) // 5))
    df['Lat_Smooth'] = df['Lat'].rolling(window=window_smooth, min_periods=1).mean()
    df['Lon_Smooth'] = df['Lon'].rolling(window=window_smooth, min_periods=1).mean()
    
    return df, hdop_removed, outliers_removed

# --- TECHNIQUE: TIME-ENCODED OCCUPANCY GRID ---
def generate_occupancy_grid(sequence, img_size=32):
    """
    Technique: Converts a GPS trajectory sequence into a Time-Encoded Image.
    
    1. Normalization: The sequence is normalized to its own bounding box (0-1).
       This makes the model invariant to global lat/lon, focusing only on the SHAPE of the drift.
       
    2. Time-Encoding: Instead of binary (0/1), pixels are valued by their time step (0.1 to 1.0).
       Older points are dim, newer points are bright. This lets the CNN "see" velocity and direction.
    """
    img = np.zeros((img_size, img_size), dtype=np.float32)
    
    # Local Normalization
    min_val = np.min(sequence, axis=0)
    max_val = np.max(sequence, axis=0)
    denom = max_val - min_val
    denom[denom == 0] = 1e-6 # Avoid div by zero
    
    norm_seq = (sequence - min_val) / denom
    
    # Scale to integer indices [0, img_size-1]
    indices = (norm_seq * (img_size - 1)).astype(int)
    
    # Draw path with Time Encoding
    num_points = len(indices)
    for i in range(num_points):
        r, c = indices[i, 0], indices[i, 1]
        
        # Image coordinates: Origin is Top-Left. 
        # Map Latitude (Y) to Row (inverse)
        row_idx = img_size - 1 - r
        col_idx = c
        
        if 0 <= row_idx < img_size and 0 <= col_idx < img_size:
            # Brightness increases with time (i)
            intensity = (i + 1) / num_points 
            img[row_idx, col_idx] = intensity
        
    return img[np.newaxis, :, :] # Add channel dimension (1, H, W)

# --- MODEL 3: LSTM (Time Series) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- MODEL 4: CNN (Computer Vision) ---
class GPS_CNN(nn.Module):
    def __init__(self):
        super(GPS_CNN, self).__init__()
        # Input: 1 x 32 x 32
        # Feature Extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) # -> 16 x 16
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # -> 8 x 8
        
        # Regression Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2) # Output: Lat, Lon
        
    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload GPS CSV", type=["csv", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    lat_col, lon_col, hdop_col = find_coordinate_columns(df_raw)

    if not lat_col or not lon_col:
        st.error(f"❌ Missing 'Lat'/'Lon' columns. Found: {list(df_raw.columns)}")
        st.stop()

    df_proc, hdop_count, out_count = preprocess_data(df_raw.copy(), lat_col, lon_col, hdop_col)
    
    # UI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", len(df_raw))
    m2.metric("Filtered (HDOP)", hdop_count)
    m3.metric("Filtered (Outlier)", out_count)
    m4.metric("Valid Rows", len(df_proc))

    # Safety Check
    RUN_DL = False
    safe_look_back = LOOK_BACK
    if len(df_proc) > LOOK_BACK + 5:
        RUN_DL = True
    elif len(df_proc) > 10:
        safe_look_back = max(2, len(df_proc) // 2)
        RUN_DL = True
        st.warning(f"⚠️ Adjusted Look-Back to {safe_look_back} due to small dataset.")
    else:
        st.error("⚠️ Data too small for Deep Learning. Showing Statistical/Kalman results only.")

    # Tabs
    tab_train, tab_viz, tab_cnn = st.tabs(["🚀 Training & Processing", "📊 Final Results", "👁️ CNN Concept"])

    # Results Containers
    if 'results' not in st.session_state:
        st.session_state.results = {}

    with tab_train:
        st.subheader("1. Mathematical Models (Deterministic)")
        
        # --- MODEL 1: STATISTICAL MEDIAN ---
        lat_med = df_proc['Lat'].median()
        lon_med = df_proc['Lon'].median()
        st.session_state.results['Median'] = (lat_med, lon_med)
        st.info(f"✅ Median Calculated: {lat_med:.6f}, {lon_med:.6f}")

        # --- MODEL 2: KALMAN FILTER ---
        kf_data = df_proc[['Lat', 'Lon']].values
        kf = KalmanFilter(
            initial_state_mean=kf_data[0],
            observation_covariance=np.eye(2)*0.0001,
            transition_covariance=np.eye(2)*1e-5
        )
        smoothed_means, _ = kf.smooth(kf_data)
        lat_kf, lon_kf = smoothed_means[-1]
        st.session_state.results['Kalman'] = (lat_kf, lon_kf)
        st.session_state.results['Kalman_Path'] = smoothed_means
        st.success(f"✅ Kalman Filter Converged: {lat_kf:.6f}, {lon_kf:.6f}")

        # --- DEEP LEARNING PREPARATION ---
        if RUN_DL:
            st.subheader("2. Deep Learning Models (Train)")
            
            # Prepare Data
            raw_vals = df_proc[['Lat_Smooth', 'Lon_Smooth']].values
            scaler = MinMaxScaler()
            scaled_vals = scaler.fit_transform(raw_vals)
            
            # Create Sequences
            X_seq, y_target = [], []
            X_img = [] # For CNN
            
            # Create data for both models
            for i in range(len(scaled_vals) - safe_look_back):
                seq = scaled_vals[i:i+safe_look_back]
                target = scaled_vals[i+safe_look_back]
                
                # Input for LSTM: Raw Sequence
                X_seq.append(seq)
                
                # Input for CNN: Generated Image
                X_img.append(generate_occupancy_grid(seq, IMG_SIZE))
                
                y_target.append(target)

            # Tensors
            X_lstm = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
            X_cnn = torch.tensor(np.array(X_img), dtype=torch.float32).to(device)
            y_t = torch.tensor(np.array(y_target), dtype=torch.float32).to(device)

            # Store last sample for visualization
            last_seq_sample = scaled_vals[-safe_look_back:]
            st.session_state.results['Last_Img'] = generate_occupancy_grid(last_seq_sample, IMG_SIZE)[0]

            col_lstm, col_cnn = st.columns(2)
            
            # --- MODEL 3: LSTM TRAINING ---
            with col_lstm:
                st.markdown("#### Model 3: LSTM")
                st.caption("Recurrent Neural Network on Time Series")
                if st.button("Train LSTM"):
                    lstm_model = LSTMModel().to(device)
                    opt = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
                    crit = nn.MSELoss()
                    
                    dataset = TensorDataset(X_lstm, y_t)
                    loader = DataLoader(dataset, batch_size=16, shuffle=True)
                    
                    prog = st.progress(0)
                    for e in range(EPOCHS):
                        lstm_model.train()
                        for bx, by in loader:
                            opt.zero_grad()
                            loss = crit(lstm_model(bx), by)
                            loss.backward()
                            opt.step()
                        prog.progress((e+1)/EPOCHS)
                    
                    # Predict
                    lstm_model.eval()
                    with torch.no_grad():
                        last_in = torch.tensor(last_seq_sample, dtype=torch.float32).unsqueeze(0).to(device)
                        pred_scaled = lstm_model(last_in).cpu().numpy()
                        pred_real = scaler.inverse_transform(pred_scaled)[0]
                        st.session_state.results['LSTM'] = (pred_real[0], pred_real[1])
                    st.success(f"LSTM Prediction: {pred_real[0]:.6f}, {pred_real[1]:.6f}")

            # --- MODEL 4: CNN TRAINING ---
            with col_cnn:
                st.markdown("#### Model 4: CNN")
                st.caption("Convolutional Network on Occupancy Grids")
                if st.button("Train CNN"):
                    cnn_model = GPS_CNN().to(device)
                    opt = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
                    crit = nn.MSELoss()
                    
                    dataset = TensorDataset(X_cnn, y_t)
                    loader = DataLoader(dataset, batch_size=16, shuffle=True)
                    
                    prog = st.progress(0)
                    for e in range(EPOCHS):
                        cnn_model.train()
                        for bx, by in loader:
                            opt.zero_grad()
                            loss = crit(cnn_model(bx), by)
                            loss.backward()
                            opt.step()
                        prog.progress((e+1)/EPOCHS)
                        
                    # Predict
                    cnn_model.eval()
                    with torch.no_grad():
                        # Generate image for the FINAL sequence
                        last_img = generate_occupancy_grid(last_seq_sample, IMG_SIZE)
                        last_in = torch.tensor(last_img[np.newaxis, ...], dtype=torch.float32).to(device)
                        pred_scaled = cnn_model(last_in).cpu().numpy()
                        pred_real = scaler.inverse_transform(pred_scaled)[0]
                        st.session_state.results['CNN'] = (pred_real[0], pred_real[1])
                    st.success(f"CNN Prediction: {pred_real[0]:.6f}, {pred_real[1]:.6f}")

    with tab_viz:
        st.subheader("Multi-Model Results Comparison")
        
        res = st.session_state.results
        if 'Median' in res:
            
            # 1. Numerical Comparison
            # Calculate distances from Median (Baseline)
            base_lat, base_lon = res['Median']
            
            data = {
                "Method": ["Statistical Median", "Kalman Filter"],
                "Latitude": [base_lat, res['Kalman'][0]],
                "Longitude": [base_lon, res['Kalman'][1]],
                "Diff from Median (m)": [0.0, haversine_distance(base_lat, base_lon, res['Kalman'][0], res['Kalman'][1])]
            }
            
            if 'LSTM' in res:
                data["Method"].append("LSTM (Time Series)")
                data["Latitude"].append(res['LSTM'][0])
                data["Longitude"].append(res['LSTM'][1])
                data["Diff from Median (m)"].append(haversine_distance(base_lat, base_lon, res['LSTM'][0], res['LSTM'][1]))
            
            if 'CNN' in res:
                data["Method"].append("CNN (Vision)")
                data["Latitude"].append(res['CNN'][0])
                data["Longitude"].append(res['CNN'][1])
                data["Diff from Median (m)"].append(haversine_distance(base_lat, base_lon, res['CNN'][0], res['CNN'][1]))

            st.dataframe(pd.DataFrame(data).style.format({"Latitude": "{:.7f}", "Longitude": "{:.7f}", "Diff from Median (m)": "{:.2f}"}))

            # 2. Map Visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Raw Data Cloud
            ax.scatter(df_proc['Lon'], df_proc['Lat'], c='gray', alpha=0.3, s=15, label='Raw GPS Data')
            
            # Kalman Path
            if 'Kalman_Path' in res:
                path = res['Kalman_Path']
                ax.plot(path[:, 1], path[:, 0], c='cyan', lw=2, alpha=0.6, label='Kalman Path')

            # Markers
            ax.scatter(base_lon, base_lat, c='green', s=150, marker='o', edgecolors='black', zorder=5, label='Median')
            ax.scatter(res['Kalman'][1], res['Kalman'][0], c='blue', s=150, marker='*', edgecolors='black', zorder=5, label='Kalman')
            
            if 'LSTM' in res:
                ax.scatter(res['LSTM'][1], res['LSTM'][0], c='purple', s=150, marker='^', edgecolors='black', zorder=5, label='LSTM')
            if 'CNN' in res:
                ax.scatter(res['CNN'][1], res['CNN'][0], c='red', s=150, marker='s', edgecolors='black', zorder=5, label='CNN')

            # Confidence Circle
            lat_std_m = df_proc['Lat'].std() * 111320
            lon_std_m = df_proc['Lon'].std() * 40075000 * np.cos(np.radians(base_lat)) / 360
            radius_deg = ((lat_std_m + lon_std_m) / 2) / 111320
            circle = Circle((base_lon, base_lat), radius_deg, color='green', fill=False, ls='--')
            ax.add_patch(circle)

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Results will appear here after processing.")

    with tab_cnn:
        st.subheader("Technique: Time-Encoded Occupancy Grid")
        st.markdown("This section visualizes how the CNN 'sees' the GPS data.")
        
        col_desc, col_img = st.columns([1, 1])
        
        with col_desc:
            st.markdown("""
            **The Specific Technique Used:**
            
            1. **Windowing:** We take the last `N` GPS points (e.g., 20 steps).
            2. **Local Normalization:** We scale the points to fit inside a 0-1 box. This removes global coordinates and leaves only the **Shape of the Drift**.
            3. **Time-Encoding:** We plot these points on a 32x32 pixel grid. 
               - **Dim pixels:** Older points in the sequence.
               - **Bright pixels:** Newer points.
               
            This creates an image that encodes **Velocity** and **Direction**, allowing the CNN to predict where the path is heading next.
            """)
        
        with col_img:
            if 'Last_Img' in st.session_state.results:
                img = st.session_state.results['Last_Img']
                fig_i, ax_i = plt.subplots(figsize=(4, 4))
                ax_i.imshow(img, cmap='gray', origin='upper')
                ax_i.set_title(f"CNN Input ({IMG_SIZE}x{IMG_SIZE})")
                ax_i.axis('off')
                st.pyplot(fig_i)
                st.caption("Brighter pixels = More recent position")
            else:
                st.info("Train the CNN to see the generated image.")