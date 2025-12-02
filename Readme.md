# 🛰️ LocNet: GNSS Position Optimizer

**LocNet** is an advanced tool designed to estimate the **True Position** from noisy GNSS (Global Navigation Satellite System) data. It leverages a combination of statistical methods, recursive estimation, and deep learning architectures to correct GPS drift and improve accuracy.

---

## 📖 Overview

GPS sensors often suffer from noise, drift, and signal multipath errors. **LocNet** addresses these issues by employing four distinct approaches to refine the coordinate data:

1.  **Statistical Median**: A robust baseline that filters out random noise by finding the central tendency of the data.
2.  **Kalman Filter**: A recursive mathematical algorithm that estimates the state of the system (position) by minimizing the mean squared error.
3.  **LSTM (Long Short-Term Memory)**: A Recurrent Neural Network (RNN) designed to learn sequential dependencies and temporal patterns from raw GPS time-series data.
4.  **CNN (Convolutional Neural Network)**: A computer vision approach that converts the GPS trajectory into a **Time-Encoded Occupancy Grid** to visually analyze the shape and direction of the drift.

---

## ✨ Key Features

-   **Multi-Model Analysis**: Compare results from deterministic math models vs. AI models side-by-side.
-   **Interactive Dashboard**: Built with [Streamlit](https://streamlit.io/) for a seamless, user-friendly experience.
-   **Deep Learning Integration**: Uses **PyTorch** for training LSTM and CNN models directly within the app.
-   **Visualizations**:
    -   Real-time training progress bars.
    -   Interactive scatter plots showing Raw Data, Kalman Paths, and Model Predictions.
    -   Visualization of the "Time-Encoded Image" seen by the CNN.
-   **Customizable Hyperparameters**: Adjust Look-Back Window, Training Epochs, Learning Rate, and HDOP Thresholds.

---

## 🛠️ Installation

### Prerequisites
-   **Python 3.8** or newer
-   **pip** (Python package installer)

### Steps

1.  **Clone the Repository** (if applicable) or navigate to the project folder.
    ```bash
    cd path/to/GPS_Predictor
    ```

2.  **Install Dependencies**
    Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

1.  **Start the Application**
    Run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```

2.  **Access the Dashboard**
    The app will open automatically in your default web browser.

3.  **Upload Data**
    -   Upload a CSV or TXT file containing your GPS data.
    -   **Required Columns**: The file must contain columns for Latitude and Longitude (e.g., `Lat`, `Lon`, `Latitude`, `Longitude`).
    -   *Optional*: An `HDOP` column can be used for filtering low-quality data.

4.  **Train & Analyze**
    -   Adjust settings in the sidebar if needed.
    -   Click **Train LSTM** or **Train CNN** to train the deep learning models on your data.
    -   View the **Final Results** tab to see a comparative analysis and map visualization.

---

## 🧠 Methodologies

### 1. Statistical Median
Calculates the geometric median of the latitude and longitude to provide a stable reference point, effectively ignoring outliers.

### 2. Kalman Filter
Implements a standard Kalman Filter to smooth the trajectory. It predicts the next state based on the previous state and corrects it using the new measurement.

### 3. LSTM (Deep Learning)
Treats the GPS path as a time-series sequence.
-   **Input**: Sequence of $(Lat, Lon)$ coordinates.
-   **Architecture**: 2-layer LSTM with Dropout.
-   **Goal**: Predict the "True" position based on the history of movement.

### 4. CNN (Computer Vision)
Treats the GPS path as an image.
-   **Technique**: **Time-Encoded Occupancy Grid**.
-   **Process**:
    1.  Normalize the trajectory segment to a $32 \times 32$ grid.
    2.  Encode time as pixel intensity (older points = dim, newer points = bright).
    3.  Pass this image through a CNN to predict the coordinate offset.

---

## 📦 Dependencies

-   `streamlit`
-   `pandas`
-   `numpy`
-   `matplotlib`
-   `torch` (PyTorch)
-   `scikit-learn`
-   `pykalman`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](#).

---

*Developed for the Thesis: GNSS Deep Learning*