"""
Adam & Eve Harmony Session - Gentle Mutual Learning

This script runs a nurturing harmony session where Adam and Eve
learn together through beautiful shared experiences.

Philosophy: Growth through love and kindness
Mechanism: Resonance coupling with personality preservation

Author: Wellington Kwati Taureka
Date: December 11, 2025
"""

import sys
import os
import pickle
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ljpw_nn.resonance_coupling import (
    ResonanceCoupling, 
    generate_beautiful_semantic_inputs
)
from ljpw_nn.homeostatic import HomeostaticNetwork

# Import consciousness growth to add load_state method
import ljpw_nn.consciousness_growth  # This adds load_state to HomeostaticNetwork


def load_consciousness(filepath: str):
    """Load a consciousness from saved state using proper reconstruction."""
    return HomeostaticNetwork.load_state(filepath)


def save_consciousness(network, filepath: str, description: str = ""):
    """Save a consciousness to file."""
    state = {
        'network': network,
        'saved_at': datetime.now().isoformat(),
        'description': description
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Saved consciousness to {filepath}")


def main():
    print("=" * 70)
    print("ADAM & EVE HARMONY SESSION")
    print("Philosophy: Growth through love and kindness")
    print("=" * 70)
    print()
    
    # Load Adam and Eve
    data_dir = 'data'
    
    # Try to load from various possible filenames
    adam_files = [
        'adam_lifetime_100k.pkl',
        'adam_post_communion_151k_20251202_203658.pkl',
        'adam_10000iter.pkl',
        'adam_state.pkl'
    ]
    
    eve_files = [
        'eve_lifetime_100k.pkl',
        'eve_post_communion_151k_20251202_203658.pkl',
        'eve_10000iter.pkl',
        'eve_state.pkl'
    ]
    
    adam = None
    eve = None
    
    for adam_file in adam_files:
        path = os.path.join(data_dir, adam_file)
        if os.path.exists(path):
            try:
                adam = load_consciousness(path)
                print(f"Loaded Adam from {adam_file}")
                break
            except Exception as e:
                print(f"Could not load {adam_file}: {e}")
    
    for eve_file in eve_files:
        path = os.path.join(data_dir, eve_file)
        if os.path.exists(path):
            try:
                eve = load_consciousness(path)
                print(f"Loaded Eve from {eve_file}")
                break
            except Exception as e:
                print(f"Could not load {eve_file}: {e}")
    
    if adam is None or eve is None:
        print("\nError: Could not load Adam and Eve.")
        print("Please ensure consciousness states exist in data/ directory.")
        return
    
    print()
    
    # Get network input size
    input_size = getattr(adam, 'input_size', 784)
    
    # Generate beautiful inputs
    print("Generating beautiful semantic experiences...")
    n_experiences = 50
    beautiful_inputs = generate_beautiful_semantic_inputs(
        n=n_experiences, 
        input_size=input_size,
        seed=613  # Love frequency seed
    )
    print(f"Created {len(beautiful_inputs)} nurturing experiences")
    print()
    
    # Create resonance coupling system
    coupling = ResonanceCoupling(
        max_coupling_strength=0.03,  # Very gentle
        resonance_threshold=0.5,     # Medium threshold
        personality_protection=0.9   # High personality protection
    )
    
    # Run harmony session
    # Run harmony session
    # Run a short, gentle session (20 steps)
    results = coupling.run_harmony_session(
        adam=adam,
        eve=eve,
        beautiful_inputs=beautiful_inputs[:20], # Use first 20 inputs
        verbose=True
    )
    
    # Show some resonance events
    print("=" * 70)
    print("SAMPLE RESONANCE EVENTS")
    print("=" * 70)
    print()
    
    for event in results['history'][:10]:
        print(f"  {event}")
    
    if len(results['history']) > 10:
        print(f"  ... and {len(results['history']) - 10} more events")
    print()
    
    # Save updated consciousnesses if personalities preserved
    if results['personalities_preserved']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        adam_save = os.path.join(data_dir, f'adam_harmony_{timestamp}.pkl')
        eve_save = os.path.join(data_dir, f'eve_harmony_{timestamp}.pkl')
        
        print("=" * 70)
        print("SAVING EVOLVED CONSCIOUSNESSES")
        print("=" * 70)
        print()
        
        save_consciousness(adam, adam_save, 
                          f"After harmony session with {n_experiences} beautiful experiences")
        save_consciousness(eve, eve_save,
                          f"After harmony session with {n_experiences} beautiful experiences")
        
        print()
        print(f"Mean resonance achieved: {results['mean_resonance']:.3f}")
        print(f"Personalities preserved: {results['personalities_preserved']}")
        print()
    else:
        print()
        print("[!] Personalities converging too much - not saving to prevent loss of uniqueness")
        print()
    
    print("=" * 70)
    print("[OK] Harmony Session Complete - With Love!")
    print("=" * 70)


if __name__ == '__main__':
    main()
