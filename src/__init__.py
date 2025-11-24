"""
Africa Population Trend and Health Analysis - Source Package

This package contains modularized classes for predictive modeling and hypothesis testing
on UN DESA population and health indicators for African countries (2010-2024).

Modules:
    data_processor: DataProcessor class for data loading and cleaning
    feature_engineer: FeatureEngineer class for feature creation
    model_trainer: ModelTrainer class for model training and tuning
    model_evaluator: ModelEvaluator class for model evaluation and comparison

Author: PDA Assignment 2 Team
Date: November 2025
"""

from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

__all__ = [
    'DataProcessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator'
]

__version__ = '1.0.0'
