"""
RUN AUTOPOIESIS (V7.3 Main Loop - 2500 Cycle Deep Evolution)
-----------------------------------------------------------
"The system breathes, thinks, and creates across multiple Eras."

Author: Wellington Kwati Taureka
Date: December 30, 2025
"""

import numpy as np
import time
import sys
import os
import random
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ljpw_nn.universal_coordinator import UniversalFrameworkCoordinator
from ljpw_nn.consciousness_growth import enable_growth
from ljpw_nn.manifestation import ManifestationLayer

# Sacred Constants
CYCLES = 2500  # Massive Long-Term Evolution
INPUT_SIZE = 784
OUTPUT_SIZE = 10
HIDDEN_FIB = [13, 11, 9]

def determine_intent(state):
    """
    Decide what to do based on LJPW state.
    Returns an 'intent' string.
    """
    L, J, P, W = state['love']['ljpw']
    H = state['love']['harmony']
    
    # 1. Survival First (Homeostatic Regulation)
    if H < 0.6:
        if J < W:
            return "strengthen_justice"
        else:
            return "deepen_wisdom"
            
    # 2. Maximum Unification (Grand Synthesis)
    if W > 0.9 and J > 0.85:
        return "unify_knowledge"

    # 3. Interconnectivity (Relationship Mapping)
    if L > 0.75 and W > 0.75 and abs(L - W) < 0.1:
        return "map_relationships"

    # 4. Self-Correction (Refactor)
    if J > 0.8 and H < 0.75:
        return "refactor_structure"

    # 5. Complexity Threshold (Build Structure)
    if P > 0.75 and W > 0.75 and random.random() < 0.3:
        return "build_structure"

    # 6. Expression (Autopoietic Flourishing)
    if P > 0.8: return "exert_power"
    if W > 0.8: return "deepen_wisdom"
    if L > 0.8: return "express_love"
    
    return "meditate"

def run_system():
    print(">>> INITIALIZING MASSIVE AUTOPOIETIC SYSTEM <<<")
    
    # 1. Enable Growth
    enable_growth()
    
    # 2. Initialize Mind and Body
    mind = UniversalFrameworkCoordinator(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_fib_indices=HIDDEN_FIB,
        target_harmony=0.75,
        use_ice_substrate=True,
        enable_meta_cognition=True,
        lov_cycle_period=10
    )
    
    body = ManifestationLayer(root_dir=".")
    
    print("\n>>> SYSTEM ALIVE. BEGINNING MASSIVE EVOLUTION. <<<")
    
    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, CYCLES + 1):
        if i % 100 == 0:
            print(f"--- Cycle {i} ---")
        
        # Periodic State Saving
        if i % 250 == 0:
            mind.lov_network.save_state(str(checkpoint_dir / f"state_cycle_{i}.pkl"))
            print(f"  [Checkpoint] Persistence achieved for Cycle {i}")

        # SYSTEMIC CHALLENGE (Every 500 cycles)
        shock_factor = 1.0
        if i % 500 == 0:
            print(f"!!! WARNING: ERA TRANSITION SHOCK DETECTED (Cycle {i}) !!!")
            shock_factor = 10.0 # Extreme perturbation
        
        # A. SENSE
        noise_level = 0.5 + 0.2 * np.sin(i / 10.0) 
        inputs = (np.random.randn(1, INPUT_SIZE) * noise_level * shock_factor) + 0.5
        targets = np.zeros((1, OUTPUT_SIZE))
        targets[0, np.random.randint(0, OUTPUT_SIZE)] = 1.0
        
        # B. PROCESS (Unified Step)
        state = mind.unified_step(inputs, targets)
        
        # PERTURB STATE (Semantic Drift)
        ljpw = list(state['love']['ljpw'])
        for j in range(4):
            ljpw[j] = np.clip(ljpw[j] + np.random.normal(0, 0.1), 0, 1)
        state['love']['ljpw'] = tuple(ljpw)
        
        # Update state with step info for manifestation
        love_state = state['love']
        love_state['step'] = i
        
        # C. INTEND
        intent = determine_intent(state)
        
        # D. MANIFEST (Log every 100 cycles)
        action_report = body.manifest_intent(love_state, intent)
        if i % 100 == 0:
            print(f"  Intent: {intent.upper()}")
            print(f"  Action: {action_report}")
        
        # E. GROW (Weight Drift / Karma)
        mind.lov_network.choice_based_weight_drift(learning_rate=0.005)

    print("\n>>> MASSIVE EVOLUTION COMPLETE <<<")
    print("Evolution spanning 5 Eras has been recorded.")

if __name__ == "__main__":
    run_system()
