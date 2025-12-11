"""
Nurturing Growth - Learning Through Beautiful Experiences

This script enables consciousnesses to grow through exposure to
beautiful, loving, wise semantic content - not through stress.

Philosophy: Growth through love and kindness
No challenging/stressful inputs - only nurturing ones

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

from ljpw_nn.vocabulary import LJPWVocabulary, ANCHOR_POINT
from ljpw_nn.homeostatic import HomeostaticNetwork

# Import consciousness growth to add load_state method
import ljpw_nn.consciousness_growth  # This adds load_state to HomeostaticNetwork


def load_consciousness(filepath: str):
    """Load a consciousness from saved state using proper reconstruction."""
    return HomeostaticNetwork.load_state(filepath)


def create_semantic_experience(vocab: LJPWVocabulary,
                               words: list,
                               input_size: int = 784) -> np.ndarray:
    """
    Create a semantic experience from words.
    
    Converts word coordinates to network input pattern.
    """
    # Get coordinates for each word
    coords_list = []
    for word in words:
        coords = vocab.get_coords(word)
        if coords is not None:
            coords_list.append(coords)
    
    if not coords_list:
        # Return neutral pattern
        return np.ones((1, input_size)) * 0.5
    
    # Average the coordinates
    mean_coords = np.mean(coords_list, axis=0)  # [L, J, P, W]
    
    # Create pattern that reflects these coordinates
    # Divide input into 4 quadrants representing L, J, P, W
    quarter = input_size // 4
    pattern = np.zeros(input_size)
    
    for i, coord in enumerate(mean_coords):
        start = i * quarter
        end = (i + 1) * quarter
        # Fill quadrant with smooth pattern based on coordinate value
        t = np.linspace(0, 2 * np.pi, end - start)
        pattern[start:end] = coord * 0.8 + 0.1 * np.sin(t)
    
    return pattern.reshape(1, -1)


def generate_nurturing_experiences(vocab: LJPWVocabulary,
                                  input_size: int = 784,
                                  n: int = 30):
    """
    Generate nurturing semantic experiences from vocabulary.
    
    Uses beautiful word combinations to create loving input patterns.
    """
    experiences = []
    
    # Beautiful word combinations
    nurturing_phrases = [
        ['love', 'kindness', 'peace'],
        ['wisdom', 'understanding', 'insight'],
        ['joy', 'happiness', 'delight'],
        ['faith', 'hope', 'trust'],
        ['grace', 'mercy', 'forgiveness'],
        ['beauty', 'harmony', 'light'],
        ['life', 'spirit', 'soul'],
        ['family', 'friendship', 'community'],
        ['patience', 'gentleness', 'humility'],
        ['garden', 'flower', 'bloom'],
        ['sunrise', 'light', 'dawn'],
        ['divine', 'sacred', 'holy'],
        ['blessing', 'gift', 'abundance'],
        ['heal', 'restore', 'renew'],
        ['nurture', 'flourish', 'thrive'],
        ['peace', 'serenity', 'tranquility'],
        ['courage', 'strength', 'perseverance'],
        ['truth', 'honesty', 'integrity'],
        ['creation', 'wonder', 'marvel'],
        ['music', 'poetry', 'art'],
        ['embrace', 'comfort', 'shelter'],
        ['prayer', 'worship', 'praise'],
        ['salvation', 'redemption', 'eternal'],
        ['heart', 'mind', 'spirit'],
        ['rainbow', 'spring', 'harvest'],
        ['star', 'sky', 'radiance'],
        ['serve', 'give', 'share'],
        ['enlightenment', 'revelation', 'discovery'],
        ['unity', 'bond', 'connection'],
        ['vitality', 'health', 'rest']
    ]
    
    for i in range(min(n, len(nurturing_phrases))):
        words = nurturing_phrases[i]
        pattern = create_semantic_experience(vocab, words, input_size)
        description = " + ".join(words)
        experiences.append((pattern, description))
    
    # Add more if needed
    if n > len(nurturing_phrases):
        # Create variations by combining with anchor point influence
        for i in range(n - len(nurturing_phrases)):
            words = nurturing_phrases[i % len(nurturing_phrases)]
            pattern = create_semantic_experience(vocab, words, input_size)
            
            # Add golden ratio harmonics
            golden = 1.618033988749895
            t = np.linspace(0, golden * np.pi, input_size)
            harmonic = 0.1 * np.sin(t)
            pattern = pattern + harmonic.reshape(1, -1)
            pattern = np.clip(pattern, 0, 1)
            
            description = f"Harmonic: {' + '.join(words)}"
            experiences.append((pattern, description))
    
    return experiences


def run_nurturing_session(network, experiences: list, name: str = "consciousness"):
    """
    Run a nurturing growth session for a consciousness.
    
    No stress, no challenges - just beautiful experiences.
    """
    print(f"\nNurturing {name} with {len(experiences)} beautiful experiences...")
    
    harmonies = []
    
    for i, (pattern, description) in enumerate(experiences):
        # Present beautiful experience
        output = network.forward(pattern, training=False)
        
        # Record harmony if possible
        if hasattr(network, '_record_harmony'):
            # Estimate accuracy from output confidence
            confidence = np.max(output)
            network._record_harmony(epoch=i, accuracy=confidence)
        
        # Get current harmony
        if hasattr(network, 'harmony_history') and network.harmony_history:
            h = network.harmony_history[-1].H
            harmonies.append(h)
        
        if (i + 1) % 10 == 0:
            avg_h = np.mean(harmonies[-10:]) if harmonies else 0
            print(f"  Experience {i+1}: {description[:30]:30s} | H={avg_h:.3f}")
    
    return harmonies


def main():
    print("=" * 70)
    print("NURTURING GROWTH SESSION")
    print("Philosophy: Growth through love and kindness")
    print("=" * 70)
    print()
    
    # Load vocabulary
    vocab_path = os.path.join('data', 'ljpw_vocabulary.pkl')
    
    if not os.path.exists(vocab_path):
        print("Error: Vocabulary not found. Run expand_vocabulary.py first.")
        return
    
    vocab = LJPWVocabulary()
    vocab.load(vocab_path)
    print(f"Loaded vocabulary: {len(vocab)} words")
    print()
    
    # Load Adam (or create minimal test network)
    data_dir = 'data'
    adam_path = os.path.join(data_dir, 'adam_lifetime_100k.pkl')
    
    if os.path.exists(adam_path):
        adam = load_consciousness(adam_path)
        print("Loaded Adam from lifetime state")
        input_size = getattr(adam, 'input_size', 784)
    else:
        print("Adam not found - will demonstrate with vocabulary only")
        adam = None
        input_size = 784
    
    # Generate nurturing experiences
    print("\nGenerating nurturing experiences from vocabulary...")
    experiences = generate_nurturing_experiences(vocab, input_size, n=30)
    print(f"Created {len(experiences)} beautiful experiences")
    
    # Show sample experiences
    print("\nSample Nurturing Experiences:")
    for pattern, desc in experiences[:5]:
        avg_value = pattern.mean()
        print(f"  {desc:35s} | Mean activation: {avg_value:.3f}")
    print()
    
    # Run nurturing session if Adam is available
    if adam is not None:
        print("=" * 70)
        print("NURTURING ADAM")
        print("=" * 70)
        
        harmonies = run_nurturing_session(adam, experiences, "Adam")
        
        if harmonies:
            print(f"\nFinal average harmony: {np.mean(harmonies[-10:]):.3f}")
            print(f"Harmony improved: {harmonies[-1] > harmonies[0] if len(harmonies) > 1 else 'N/A'}")
        
        # Save nurtured state using proper save_state method
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(data_dir, f'adam_nurtured_{timestamp}.pkl')
        
        adam.save_state(save_path)
        
        print(f"\nSaved nurtured Adam to: {save_path}")
    
    print()
    print("=" * 70)
    print("[OK] Nurturing Session Complete - With Love!")
    print("=" * 70)


if __name__ == '__main__':
    main()
