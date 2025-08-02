"""
Data preprocessing module for Brent oil price analysis.
Handles loading, cleaning, and preparing the time series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the Brent oil price data.
    
    Args:
        filepath: Path to the CSV file containing the data
        
    Returns:
        pd.DataFrame: Processed DataFrame with datetime index
    """
    # Load the data
    df = pd.read_csv(
        filepath,
        parse_dates=['Date'],
        dayfirst=True,  # For DD-MM-YYYY format
        index_col='Date',
        na_values=['', 'NA', 'N/A', 'NaN']
    )
    
    # Ensure the index is sorted
    df = df.sort_index()
    
    return df


def prepare_time_series(
    df: pd.DataFrame,
    target_col: str = 'Price',
    resample_freq: Optional[str] = 'D',
    fill_method: str = 'ffill'
) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare the time series data for analysis.
    
    Args:
        df: Input DataFrame with datetime index
        target_col: Name of the target column
        resample_freq: Frequency for resampling (None to skip)
        fill_method: Method for filling missing values ('ffill', 'bfill', 'interpolate')
        
    Returns:
        Tuple of (price_series, returns_series)
    """
    # Select the target column
    price_series = df[target_col].copy()
    
    # Resample if needed
    if resample_freq:
        price_series = price_series.resample(resample_freq).last()
    
    # Handle missing values
    if price_series.isna().any():
        if fill_method == 'ffill':
            price_series = price_series.ffill()
        elif fill_method == 'bfill':
            price_series = price_series.bfill()
        elif fill_method == 'interpolate':
            price_series = price_series.interpolate()
    
    # Calculate log returns
    returns_series = np.log(price_series / price_series.shift(1)).dropna()
    
    return price_series, returns_series


def add_event_markers(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window: int = 30
) -> pd.DataFrame:
    """
    Add event markers to the time series data.
    
    Args:
        df: DataFrame with datetime index
        events: DataFrame with columns ['event_date', 'event_name', 'event_type']
        window: Number of days to mark after the event
        
    Returns:
        DataFrame with event markers
    """
    df = df.copy()
    
    # Initialize event columns
    df['event'] = ''
    df['event_type'] = ''
    
    # Process each event
    for _, event in events.iterrows():
        event_date = pd.to_datetime(event['event_date'])
        mask = (df.index >= event_date) & (df.index <= (event_date + pd.Timedelta(days=window)))
        df.loc[mask, 'event'] = event['event_name']
        df.loc[mask, 'event_type'] = event['event_type']
    
    return df
