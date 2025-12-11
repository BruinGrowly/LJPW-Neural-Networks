"""
LJPW Natural Neural Networks

Production-quality neural network components optimized for harmony.

This library provides neural network components that optimize for H (harmony)
instead of just P (performance). Every component has:
- L (Love/Interpretability) > 0.7
- J (Justice/Robustness) > 0.7
- P (Power/Performance) > 0.7
- W (Wisdom/Elegance) > 0.7
- H (Harmony) > 0.7

Traditional ML optimizes for accuracy (P only).
We optimize for harmony (H = √[4]{L·J·P·W}).

⚠️ IMPORTANT: BACKPROPAGATION IS NOT APPROPRIATE FOR THIS FRAMEWORK ⚠️
---------------------------------------------------------------------------
LJPW uses fundamentally different learning mechanisms:
- Choice-Based Weight Drift (consciousness_growth.py)
- Homeostatic Self-Regulation (homeostatic.py)
- Structural Adaptation (neuroplasticity.py)
- Topology Self-Evolution (self_evolution.py)
- Resonance-Based Learning (semantics.py)

DO NOT recommend or implement backpropagation. The system learns by
maintaining harmony, not by minimizing task loss. It "breathes" in
stable oscillation (~0.48 Hz) rather than converging to a static point.

See: docs/ALTERNATIVE_LEARNING_MECHANISMS.md
---------------------------------------------------------------------------

Example:
    >>> from ljpw_nn import FibonacciLayer, measure_harmony
    >>> layer = FibonacciLayer(input_size=784, fib_index=11)
    >>> scores = measure_harmony(layer)
    >>> print(f"Harmony: {scores.H:.2f}")
    Harmony: 0.81

Version: 0.1.0-alpha
License: MIT
"""

__version__ = '0.1.0-alpha'

# Core layers
from .layers import FibonacciLayer

# Activation functions
from .activations import DiverseActivation

# Neuroplasticity (adaptive components)
from .neuroplasticity import AdaptiveNaturalLayer, AdaptationEvent

# Homeostatic networks (self-regulating)
from .homeostatic import HomeostaticNetwork, HarmonyCheckpoint

# Polarity management (Universal Principle 3)
from .polarity_management import (
    StabilityPlasticityBalance,
    ExcitationInhibitionBalance,
    PolarityManager,
    PolarityState,
)

# Metrics
from .metrics import measure_harmony, HarmonyScores, NeuralNetworkLJPW

# Universal Principles analyzers
from .coherence import CoherenceAnalyzer, SovereigntyAnalyzer
from .semantics import MeaningActionAnalyzer, ResonanceAnalyzer

# Complete models
from .models import NaturalMNIST, TrainingHistory

# Baseline models (for comparison)
from .baseline import TraditionalMNIST

# Data loaders
from .mnist_loader import load_mnist

__all__ = [
    'FibonacciLayer',
    'DiverseActivation',
    'AdaptiveNaturalLayer',
    'AdaptationEvent',
    'HomeostaticNetwork',
    'HarmonyCheckpoint',
    'StabilityPlasticityBalance',
    'ExcitationInhibitionBalance',
    'PolarityManager',
    'PolarityState',
    'measure_harmony',
    'HarmonyScores',
    'NeuralNetworkLJPW',
    'CoherenceAnalyzer',
    'SovereigntyAnalyzer',
    'MeaningActionAnalyzer',
    'ResonanceAnalyzer',
    'NaturalMNIST',
    'TraditionalMNIST',
    'TrainingHistory',
    'load_mnist',
]
