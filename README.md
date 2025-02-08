# Stellar Navigator

Stellar Navigator is an AI-powered space mission control dashboard built with Streamlit. The app integrates live space data from NASA and ISS APIs, visualizes real star data from the HYG Stellar Database, and leverages machine learning for fuel optimization predictions and anomaly detection in mission logs.

## Features

- **Live Space Data**: 
  - Real-time International Space Station (ISS) tracker using a 3D globe.
  - NASA Astronomy Picture of the Day (APOD) display.
- **Interactive Star Field Explorer**: 
  - 3D starfield visualization based on the HYG v4.1 catalog.
  - Adjustable slider to filter stars by maximum magnitude.
- **AI Fuel Optimization System**:
  - Predicts fuel usage based on mission parameters (distance, payload, engine type).
  - Visualizes predicted fuel savings with a gauge and fuel usage trend line.
  - Stores prediction history in session state.
- **Mission Log Anomaly Detection**:
  - Uses a transformer model to classify mission logs as normal or anomalous.

## Prerequisites

- Python 3.7 or later
- [Streamlit](https://streamlit.io/)
- Other required packages listed in `requirements.txt`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/stellar-navigator.git
   cd stellar-navigator
Create a Virtual Environment and Activate It:

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

bash
Copy
pip install -r requirements.txt
If you do not have a requirements.txt, install the packages manually:

bash
Copy
pip install streamlit requests pandas numpy plotly xgboost joblib transformers python-dotenv
Set Up Environment Variables:

Create a .env file in the project root and add your NASA API key:

env
Copy
NASA_API_KEY=YOUR_NASA_API_KEY
Usage
Disable File Watching (Optional):

To avoid issues with Torch file watching, create a .streamlit/config.toml file with the following content:

toml
Copy
[server]
fileWatcherType = "none"
Run the Application:

bash
Copy
streamlit run app.py
Explore the Dashboard:

Open your browser and navigate to http://localhost:8501 to view the app.

Project Structure
bash
Copy
stellar-navigator/
├── app.py            # Main application file
├── .env              # Environment variables (e.g., NASA_API_KEY)
├── .streamlit/
│   └── config.toml   # Streamlit configuration (optional)
├── README.md         # Project documentation
└── requirements.txt  # List of Python dependencies
License
This project is licensed under the MIT License.

Acknowledgments
NASA API
International Space Station API
HYG Stellar Database
Streamlit
Transformers by Hugging Face
pgsql
Copy

---

## Final Notes

- **Star Data URL Update**: The key fix for the 404 error was updating the star data URL to point to `hyg_v41.csv` in the `hyg/CURRENT/` directory of the HYG-Database repository.
- **Streamlit Configuration**: The optional `.streamlit/config.toml` file can help avoid Torch-related file watching errors.

With these changes, your app should load the star data without error and you have a comprehensive README to guide installation and usage. Enjoy exploring Stellar Navigator!





