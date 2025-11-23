"""
Data loading and validation utilities for REIT/InvIT asset data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class AssetModel(BaseModel):
    """Pydantic model for asset data validation."""
    asset_id: str
    name: str
    asset_type: str
    base_rent_or_tariff: float = Field(gt=0, description="Starting rent/tariff value")
    current_occupancy_or_utilization: float = Field(ge=0, le=1, description="Current occupancy % (0-1)")
    rent_tariff_growth_mu: float = Field(description="Expected annual growth rate")
    rent_tariff_growth_sigma: float = Field(ge=0, description="Growth volatility")
    operating_expense_ratio: float = Field(ge=0, le=1, description="Operating expenses as % of revenue")
    debt_ratio: float = Field(ge=0, le=1, description="Debt as % of NAV")
    debt_cost: float = Field(ge=0, description="Annual interest rate on debt")
    payout_ratio: float = Field(ge=0, le=1, description="Distribution payout % of DCF")
    current_nav: float = Field(gt=0, description="Current Net Asset Value")
    beta: float = Field(description="Beta vs REIT/Infrastructure index")


def load_assets(file_path: str) -> pd.DataFrame:
    """
    Load asset data from CSV file.
    
    Args:
        file_path: Path to CSV file containing asset data
        
    Returns:
        DataFrame with validated asset data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data validation fails
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Asset data file not found: {file_path}")
    
    # Validate required columns
    required_columns = [
        'asset_id', 'name', 'asset_type', 'base_rent_or_tariff',
        'current_occupancy_or_utilization', 'rent_tariff_growth_mu',
        'rent_tariff_growth_sigma', 'operating_expense_ratio',
        'debt_ratio', 'debt_cost', 'payout_ratio', 'current_nav', 'beta'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Fill missing values with documented assumptions
    df = df.fillna({
        'operating_expense_ratio': 0.30,  # Default 30% opex ratio
        'debt_ratio': 0.50,  # Default 50% debt ratio
        'debt_cost': 0.06,  # Default 6% interest rate
        'payout_ratio': 0.90,  # Default 90% payout ratio
        'beta': 1.0,  # Default beta of 1.0
    })
    
    # Validate data using Pydantic
    validated_assets = []
    for _, row in df.iterrows():
        try:
            asset = AssetModel(**row.to_dict())
            validated_assets.append(asset.model_dump())
        except Exception as e:
            raise ValueError(f"Validation error for asset {row.get('asset_id', 'unknown')}: {e}")
    
    return pd.DataFrame(validated_assets)


def validate_asset_data(df: pd.DataFrame) -> bool:
    """
    Validate asset DataFrame for correctness.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if df.empty:
        raise ValueError("Asset DataFrame is empty")
    
    # Check for duplicate asset IDs
    if df['asset_id'].duplicated().any():
        raise ValueError("Duplicate asset IDs found")
    
    # Check data ranges
    if (df['current_occupancy_or_utilization'] < 0).any() or \
       (df['current_occupancy_or_utilization'] > 1).any():
        raise ValueError("Occupancy/utilization must be between 0 and 1")
    
    if (df['operating_expense_ratio'] < 0).any() or \
       (df['operating_expense_ratio'] > 1).any():
        raise ValueError("Operating expense ratio must be between 0 and 1")
    
    if (df['debt_ratio'] < 0).any() or (df['debt_ratio'] > 1).any():
        raise ValueError("Debt ratio must be between 0 and 1")
    
    if (df['payout_ratio'] < 0).any() or (df['payout_ratio'] > 1).any():
        raise ValueError("Payout ratio must be between 0 and 1")
    
    return True

