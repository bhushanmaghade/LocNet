LocNet: GPS Position Optimizer

This project helps you find the true location from noisy GPS data. GPS sensors often drift or show errors. This tool uses four different methods to fix those errors and find the accurate position.

The Four Methods Used:

1. Median: Finds the center point of all the data to ignore random errors.

2. Kalman Filter: A mathematical formula that smooths out the path to reduce noise.

3. LSTM: A smart computer program that learns patterns from the numbers to predict the correct location.

4. CNN: Turns the GPS path into an image and uses computer vision to see the direction of the drift.

How to Start

1. Install the Libraries
Open your command prompt or terminal and run this command:

pip install -r requirements.txt

2. Run the Application
Type this command to start the tool:

streamlit run app.py

3. Use the Tool

A web page will open at http://localhost:8501.

Upload your CSV file. It must have columns named Lat and Lon.

Click the Train button to see the results.

Requirements

Python 3.8 or newer

The libraries listed in the requirements.txt file (like Streamlit, PyTorch, and Pandas).