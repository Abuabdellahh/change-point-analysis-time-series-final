"""
Bayesian Change Point Detection for Brent Oil Price Analysis

This module implements a Bayesian change point detection model using PyMC3.
It identifies structural breaks in the time series data where the underlying
data generating process changes.
"""

import numpy as np
import pymc3 as pm
import arviz as az
import pandas as pd
from typing import Tuple, Dict, Optional, List
import theano.tensor as tt


class BayesianChangePointModel:
    """Bayesian Change Point Model for detecting structural breaks in time series."""
    
    def __init__(self, n_changepoints: int = 5):
        """
        Initialize the Bayesian Change Point Model.
        
        Args:
            n_changepoints: Maximum number of change points to consider
        """
        self.n_changepoints = n_changepoints
        self.model = None
        self.trace = None
        self.data = None
        self.results = None
    
    def build_model(self, data: np.ndarray) -> pm.Model:
        """
        Build the PyMC3 model for change point detection.
        
        Args:
            data: 1D numpy array of time series data
            
        Returns:
            PyMC3 model
        """
        self.data = data
        n_obs = len(data)
        
        with pm.Model() as model:
            # Priors for change points (uniform over time)
            tau = pm.DiscreteUniform(
                'tau', 
                lower=0, 
                upper=n_obs-1, 
                shape=self.n_changepoints
            )
            
            # Sort the change points to ensure they're in order
            tau_sorted = pm.Deterministic('tau_sorted', tt.sort(tau))
            
            # Priors for segment means and standard deviations
            mu = pm.Normal('mu', mu=0, sigma=10, shape=self.n_changepoints + 1)
            sigma = pm.HalfNormal('sigma', sigma=1, shape=self.n_changepoints + 1)
            
            # Create indicator variables for which segment each observation belongs to
            segment = np.zeros(n_obs, dtype=np.int32)
            for i in range(n_obs):
                # Find the first change point after the current observation
                cp = np.searchsorted(tau_sorted, i)
                segment[i] = np.clip(cp, 0, self.n_changepoints)
            
            # Likelihood
            likelihood = pm.Normal(
                'likelihood',
                mu=mu[segment],
                sigma=sigma[segment],
                observed=data
            )
            
            self.model = model
            return model
    
    def fit(
        self, 
        data: np.ndarray, 
        draws: int = 2000, 
        tune: int = 1000, 
        target_accept: float = 0.9,
        **kwargs
    ) -> az.InferenceData:
        """
        Fit the Bayesian change point model to the data.
        
        Args:
            data: 1D numpy array of time series data
            draws: Number of posterior samples to draw
            tune: Number of tuning samples
            target_accept: Target acceptance probability for NUTS
            **kwargs: Additional arguments to pm.sample()
            
        Returns:
            ArviZ InferenceData object with sampling results
        """
        if self.model is None:
            self.build_model(data)
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                return_inferencedata=True,
                **kwargs
            )
        
        return self.trace
    
    def get_change_points(self, threshold: float = 0.5) -> List[Dict]:
        """
        Extract change points from the posterior distribution.
        
        Args:
            threshold: Probability threshold for considering a change point
            
        Returns:
            List of dictionaries with change point information
        """
        if self.trace is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        # Get posterior samples of change points
        tau_samples = self.trace.posterior['tau_sorted'].values
        n_chains, n_draws, n_cp = tau_samples.shape
        
        # Flatten the samples across chains
        tau_flat = tau_samples.reshape(-1, n_cp)
        
        # Count occurrences of each change point
        cp_counts = []
        for cp_idx in range(n_cp):
            unique, counts = np.unique(tau_flat[:, cp_idx], return_counts=True)
            cp_counts.append(dict(zip(unique, counts)))
        
        # Calculate probabilities and filter by threshold
        change_points = []
        for cp_idx, counts in enumerate(cp_counts):
            for tau, count in counts.items():
                prob = count / (n_chains * n_draws)
                if prob >= threshold:
                    change_points.append({
                        'cp_index': cp_idx,
                        'time_index': int(tau),
                        'probability': prob
                    })
        
        # Sort by time index
        change_points.sort(key=lambda x: x['time_index'])
        
        # Add segment information
        if len(change_points) > 0:
            change_points[0]['segment_start'] = 0
            for i in range(1, len(change_points)):
                change_points[i-1]['segment_end'] = change_points[i]['time_index']
                change_points[i]['segment_start'] = change_points[i-1]['time_index']
            change_points[-1]['segment_end'] = len(self.data) - 1
        
        self.results = change_points
        return change_points
    
    def get_segment_statistics(self) -> pd.DataFrame:
        """
        Calculate statistics for each segment between change points.
        
        Returns:
            DataFrame with segment statistics
        """
        if self.results is None or len(self.results) == 0:
            return pd.DataFrame()
        
        segments = []
        for i, cp in enumerate(self.results):
            start = cp['segment_start']
            end = cp['segment_end']
            segment_data = self.data[start:end+1]
            
            segments.append({
                'segment_id': i,
                'start_index': start,
                'end_index': end,
                'mean': float(np.mean(segment_data)),
                'std': float(np.std(segment_data)),
                'length': end - start + 1
            })
        
        return pd.DataFrame(segments)
