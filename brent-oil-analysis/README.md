# Brent Oil Price Change Point Analysis

This project analyzes Brent crude oil prices to identify significant change points and their potential correlation with major geopolitical and economic events. The analysis uses Bayesian change point detection to identify structural breaks in the time series data.

## Project Structure

```
├── data/                      # Data storage
├── notebooks/                 # Jupyter notebooks for analysis
├── src/                       # Source code
│   ├── data/                  # Data processing
│   ├── models/                # Change point models
│   ├── visualization/         # Visualization utilities
│   └── api/                   # Flask API
├── frontend/                  # React frontend
├── tests/                     # Unit and integration tests
└── docs/                      # Documentation
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   cd frontend
   npm install
   ```

## Usage

1. Run the Jupyter notebooks in the `notebooks/` directory for data exploration and analysis
2. Start the Flask API:
   ```
   python src/api/app.py
   ```
3. Start the React frontend:
   ```
   cd frontend
   npm start
   ```

## Data

Place your Brent oil price data in `data/raw/` as `brent_prices.csv` with columns:
- `Date`: Date in YYYY-MM-DD format
- `Price`: Price in USD

## License

MIT
