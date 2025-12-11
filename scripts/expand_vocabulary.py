"""
Vocabulary Expansion with Positive Concepts

This script expands the LJPW vocabulary with beautiful, nurturing words
following the philosophy of growth through love and kindness.

Categories:
- Virtues (patience, kindness, gentleness)
- Positive emotions (joy, peace, gratitude)
- Relationships (friendship, family, trust)
- Nature (garden, sunrise, river)
- Growth (bloom, flourish, nurture)

Author: Wellington Kwati Taureka
Date: December 11, 2025
Philosophy: Training with love and kindness
"""

import sys
import os
import json
import pickle
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ljpw_nn.vocabulary import LJPWVocabulary, ANCHOR_POINT, NATURAL_EQUILIBRIUM


# =============================================================================
# POSITIVE WORD DEFINITIONS
# =============================================================================
# Coordinates are [L, J, P, W] where each dimension ranges 0-1
# L = Love/Compassion, J = Justice/Righteousness, P = Power/Strength, W = Wisdom
#
# These coordinates are carefully crafted to reflect the semantic meaning:
# - High L words emphasize love, compassion, care
# - High J words emphasize fairness, righteousness, order
# - High P words emphasize strength, action, capability
# - High W words emphasize understanding, knowledge, insight

POSITIVE_WORDS = {
    # =========================================================================
    # VIRTUES - Moral excellence and goodness
    # =========================================================================
    'patience': [0.78, 0.72, 0.45, 0.85],      # Love + Wisdom dominant
    'kindness': [0.92, 0.68, 0.35, 0.72],      # Very high Love
    'gentleness': [0.88, 0.58, 0.28, 0.75],    # High Love, low Power
    'faithfulness': [0.82, 0.91, 0.65, 0.78],  # High Love + Justice
    'humility': [0.85, 0.75, 0.32, 0.88],      # High Love + Wisdom
    'generosity': [0.90, 0.73, 0.55, 0.70],    # Very high Love
    'honesty': [0.72, 0.94, 0.58, 0.82],       # Very high Justice
    'integrity': [0.75, 0.95, 0.68, 0.85],     # Very high Justice + Wisdom
    'sincerity': [0.82, 0.88, 0.52, 0.78],     # High Love + Justice
    'modesty': [0.80, 0.72, 0.28, 0.82],       # High Love, low Power
    'diligence': [0.65, 0.82, 0.78, 0.80],     # Balanced, slightly higher J+P
    'perseverance': [0.68, 0.78, 0.85, 0.82],  # High Power + Wisdom
    'temperance': [0.72, 0.85, 0.45, 0.90],    # High Justice + Wisdom
    'prudence': [0.65, 0.88, 0.55, 0.92],      # Very high Wisdom
    'fortitude': [0.62, 0.80, 0.88, 0.78],     # High Power
    'charity': [0.95, 0.72, 0.48, 0.68],       # Very high Love
    'mercy': [0.93, 0.65, 0.42, 0.75],         # Very high Love
    'grace': [0.90, 0.75, 0.55, 0.88],         # High Love + Wisdom
    'virtue': [0.82, 0.85, 0.62, 0.85],        # Balanced high
    'goodness': [0.88, 0.82, 0.58, 0.78],      # High Love + Justice
    
    # =========================================================================
    # POSITIVE EMOTIONS - States of well-being
    # =========================================================================
    'joy': [0.92, 0.65, 0.72, 0.70],           # Very high Love
    'peace': [0.88, 0.82, 0.35, 0.85],         # High Love + Wisdom, low Power
    'contentment': [0.85, 0.75, 0.42, 0.88],   # High Love + Wisdom
    'gratitude': [0.90, 0.78, 0.45, 0.82],     # Very high Love
    'serenity': [0.85, 0.78, 0.32, 0.90],      # High L+W, very low Power
    'hope': [0.82, 0.72, 0.68, 0.78],          # Balanced, slightly higher Love
    'delight': [0.88, 0.62, 0.65, 0.68],       # High Love
    'bliss': [0.92, 0.58, 0.55, 0.75],         # Very high Love
    'happiness': [0.88, 0.68, 0.62, 0.72],     # High Love
    'comfort': [0.85, 0.70, 0.45, 0.78],       # High Love
    'tranquility': [0.82, 0.78, 0.28, 0.88],   # High L+W, very low Power
    'cheerfulness': [0.85, 0.62, 0.68, 0.65],  # High Love
    'warmth': [0.90, 0.58, 0.52, 0.68],        # Very high Love
    'tenderness': [0.92, 0.55, 0.35, 0.72],    # Very high Love, low Power
    'affection': [0.91, 0.60, 0.42, 0.68],     # Very high Love
    
    # =========================================================================
    # RELATIONSHIPS - Connection and community
    # =========================================================================
    'friendship': [0.88, 0.75, 0.52, 0.72],    # High Love
    'family': [0.90, 0.78, 0.58, 0.75],        # Very high Love
    'community': [0.82, 0.80, 0.62, 0.75],     # High Love + Justice
    'trust': [0.85, 0.88, 0.55, 0.82],         # High Love + Justice
    'loyalty': [0.80, 0.90, 0.65, 0.75],       # High Justice
    'devotion': [0.88, 0.82, 0.62, 0.72],      # High Love + Justice
    'unity': [0.85, 0.82, 0.68, 0.80],         # Balanced high
    'harmony': [0.88, 0.85, 0.55, 0.90],       # High L+J+W
    'fellowship': [0.85, 0.78, 0.55, 0.72],    # High Love
    'companionship': [0.87, 0.72, 0.52, 0.70], # High Love
    'belonging': [0.85, 0.75, 0.48, 0.72],     # High Love
    'bond': [0.82, 0.78, 0.58, 0.70],          # High Love + Justice
    'connection': [0.80, 0.72, 0.55, 0.75],    # Balanced
    'intimacy': [0.90, 0.65, 0.48, 0.72],      # Very high Love
    'support': [0.85, 0.78, 0.62, 0.75],       # High Love
    
    # =========================================================================
    # NATURE - Beauty in creation
    # =========================================================================
    'garden': [0.82, 0.72, 0.55, 0.78],        # High Love (Eden symbolism)
    'sunrise': [0.78, 0.75, 0.65, 0.80],       # Balanced, hope symbolism
    'river': [0.72, 0.75, 0.68, 0.78],         # Balanced, life flow
    'mountain': [0.65, 0.80, 0.85, 0.82],      # High Power + Wisdom
    'forest': [0.75, 0.72, 0.68, 0.80],        # Balanced, wisdom symbolism
    'meadow': [0.82, 0.68, 0.52, 0.75],        # High Love, peaceful
    'ocean': [0.72, 0.75, 0.88, 0.82],         # High Power, vast
    'sky': [0.70, 0.72, 0.75, 0.85],           # High Wisdom, transcendent
    'rain': [0.75, 0.70, 0.62, 0.72],          # Balanced, blessing symbolism
    'flower': [0.88, 0.62, 0.45, 0.75],        # High Love, beauty
    'tree': [0.75, 0.78, 0.72, 0.85],          # Balanced, growth + wisdom
    'star': [0.72, 0.75, 0.68, 0.88],          # High Wisdom, guidance
    'moon': [0.78, 0.72, 0.55, 0.85],          # High Wisdom, reflection
    'sunlight': [0.82, 0.78, 0.72, 0.80],      # High Love, illumination
    'breeze': [0.80, 0.68, 0.45, 0.75],        # High Love, gentle
    'rainbow': [0.85, 0.78, 0.62, 0.80],       # High Love, promise symbolism
    'spring': [0.85, 0.72, 0.68, 0.78],        # High Love, renewal
    'harvest': [0.78, 0.82, 0.75, 0.80],       # Balanced, abundance
    
    # =========================================================================
    # GROWTH - Flourishing and development
    # =========================================================================
    'bloom': [0.85, 0.68, 0.65, 0.78],         # High Love, growth
    'flourish': [0.82, 0.75, 0.72, 0.80],      # Balanced high
    'nurture': [0.92, 0.72, 0.55, 0.78],       # Very high Love
    'cultivate': [0.78, 0.80, 0.68, 0.82],     # High Justice + Wisdom
    'thrive': [0.80, 0.78, 0.78, 0.78],        # Perfectly balanced
    'blossom': [0.88, 0.65, 0.62, 0.75],       # High Love
    'develop': [0.72, 0.78, 0.72, 0.82],       # Balanced, wisdom slight edge
    'mature': [0.75, 0.82, 0.68, 0.88],        # High Justice + Wisdom
    'evolve': [0.70, 0.75, 0.72, 0.85],        # High Wisdom
    'progress': [0.68, 0.78, 0.80, 0.82],      # High Power + Wisdom
    'advance': [0.65, 0.75, 0.82, 0.78],       # High Power
    'improve': [0.72, 0.80, 0.75, 0.82],       # Balanced high
    'enrich': [0.82, 0.75, 0.62, 0.80],        # High Love + Wisdom
    'strengthen': [0.68, 0.78, 0.88, 0.75],    # High Power
    'deepen': [0.78, 0.72, 0.55, 0.88],        # High Wisdom
    
    # =========================================================================
    # WISDOM AND UNDERSTANDING
    # =========================================================================
    'insight': [0.65, 0.78, 0.55, 0.92],       # Very high Wisdom
    'understanding': [0.78, 0.80, 0.52, 0.90], # High Love + Wisdom
    'clarity': [0.72, 0.82, 0.58, 0.88],       # High Justice + Wisdom
    'discernment': [0.68, 0.85, 0.55, 0.90],   # High Justice + Wisdom
    'perception': [0.65, 0.75, 0.55, 0.88],    # High Wisdom
    'awareness': [0.72, 0.75, 0.52, 0.85],     # High Wisdom
    'enlightenment': [0.80, 0.78, 0.55, 0.95], # Very high Wisdom
    'knowledge': [0.62, 0.78, 0.62, 0.88],     # High Wisdom
    'learning': [0.75, 0.75, 0.62, 0.85],      # Balanced, high Wisdom
    'discovery': [0.72, 0.72, 0.68, 0.85],     # High Wisdom
    'revelation': [0.78, 0.82, 0.62, 0.88],    # High Justice + Wisdom
    'truth': [0.72, 0.95, 0.58, 0.88],         # Very high Justice
    'guidance': [0.80, 0.82, 0.58, 0.88],      # High L+J+W
    'counsel': [0.78, 0.85, 0.52, 0.90],       # High Justice + Wisdom
    
    # =========================================================================
    # FAITH AND SPIRITUALITY
    # =========================================================================
    'faith': [0.85, 0.82, 0.68, 0.85],         # Balanced high
    'hope': [0.82, 0.72, 0.68, 0.78],          # High Love
    'belief': [0.78, 0.80, 0.62, 0.82],        # Balanced
    'devotion': [0.88, 0.82, 0.62, 0.72],      # High Love + Justice
    'prayer': [0.88, 0.78, 0.52, 0.85],        # High Love + Wisdom
    'worship': [0.90, 0.85, 0.55, 0.82],       # Very high Love + Justice
    'praise': [0.88, 0.78, 0.62, 0.75],        # High Love
    'blessing': [0.92, 0.78, 0.62, 0.80],      # Very high Love
    'miracle': [0.85, 0.78, 0.82, 0.85],       # Balanced high
    'sacred': [0.85, 0.90, 0.65, 0.88],        # High Justice + Wisdom
    'holy': [0.88, 0.92, 0.68, 0.90],          # Very high J+W
    'divine': [0.90, 0.90, 0.85, 0.92],        # Very high all dimensions
    'eternal': [0.82, 0.85, 0.78, 0.90],       # High all dimensions
    'salvation': [0.92, 0.88, 0.75, 0.85],     # Very high Love + Justice
    'redemption': [0.90, 0.85, 0.72, 0.82],    # Very high Love
    'forgiveness': [0.95, 0.78, 0.48, 0.82],   # Very high Love
    'repentance': [0.82, 0.88, 0.45, 0.85],    # High Justice + Wisdom
    
    # =========================================================================
    # ACTIONS OF LOVE
    # =========================================================================
    'help': [0.88, 0.75, 0.65, 0.72],          # High Love
    'care': [0.92, 0.70, 0.52, 0.75],          # Very high Love
    'serve': [0.88, 0.82, 0.62, 0.72],         # High Love + Justice
    'give': [0.90, 0.75, 0.55, 0.68],          # Very high Love
    'share': [0.88, 0.78, 0.52, 0.72],         # High Love
    'embrace': [0.92, 0.65, 0.55, 0.70],       # Very high Love
    'comfort': [0.90, 0.70, 0.48, 0.78],       # Very high Love
    'encourage': [0.85, 0.75, 0.68, 0.78],     # High Love
    'inspire': [0.82, 0.72, 0.72, 0.85],       # High Love + Wisdom
    'uplift': [0.88, 0.72, 0.68, 0.78],        # High Love
    'heal': [0.90, 0.78, 0.62, 0.82],          # Very high Love
    'restore': [0.85, 0.82, 0.68, 0.80],       # High Love + Justice
    'renew': [0.82, 0.78, 0.65, 0.82],         # Balanced
    'protect': [0.82, 0.88, 0.78, 0.75],       # High Justice + Power
    'shelter': [0.85, 0.78, 0.62, 0.72],       # High Love
    
    # =========================================================================
    # BEAUTY AND ART
    # =========================================================================
    'beauty': [0.85, 0.72, 0.55, 0.85],        # High Love + Wisdom
    'art': [0.78, 0.68, 0.65, 0.88],           # High Wisdom
    'music': [0.85, 0.70, 0.62, 0.82],         # High Love + Wisdom
    'poetry': [0.82, 0.72, 0.52, 0.88],        # High Love + Wisdom
    'song': [0.88, 0.68, 0.62, 0.78],          # High Love
    'dance': [0.82, 0.65, 0.72, 0.75],         # High Love + Power
    'creation': [0.80, 0.78, 0.82, 0.88],      # Balanced high
    'imagination': [0.75, 0.65, 0.68, 0.90],   # High Wisdom
    'wonder': [0.82, 0.68, 0.62, 0.85],        # High Love + Wisdom
    'marvel': [0.80, 0.72, 0.65, 0.82],        # Balanced
    'splendor': [0.82, 0.78, 0.72, 0.85],      # Balanced high
    'glory': [0.85, 0.85, 0.82, 0.88],         # Very high all
    'majesty': [0.78, 0.85, 0.88, 0.88],       # High Power + Justice + Wisdom
    
    # =========================================================================
    # LIFE AND VITALITY
    # =========================================================================
    'life': [0.88, 0.78, 0.82, 0.85],          # High all dimensions
    'breath': [0.82, 0.72, 0.68, 0.78],        # Balanced
    'spirit': [0.85, 0.80, 0.72, 0.90],        # High Wisdom + Love
    'soul': [0.88, 0.82, 0.68, 0.88],          # High all dimensions
    'heart': [0.92, 0.72, 0.58, 0.75],         # Very high Love
    'mind': [0.68, 0.78, 0.65, 0.92],          # Very high Wisdom
    'body': [0.72, 0.75, 0.85, 0.68],          # High Power
    'health': [0.80, 0.78, 0.82, 0.78],        # Balanced
    'vitality': [0.78, 0.75, 0.88, 0.75],      # High Power
    'energy': [0.72, 0.70, 0.90, 0.72],        # Very high Power
    'strength': [0.65, 0.78, 0.92, 0.75],      # Very high Power
    'rest': [0.82, 0.72, 0.32, 0.80],          # High Love, low Power
    'sleep': [0.78, 0.70, 0.28, 0.75],         # High Love, low Power
    'dream': [0.80, 0.65, 0.52, 0.85],         # High Love + Wisdom
    
    # =========================================================================
    # LIGHT AND HOPE
    # =========================================================================
    'light': [0.85, 0.82, 0.72, 0.88],         # High all dimensions
    'dawn': [0.82, 0.78, 0.68, 0.82],          # Balanced high
    'shine': [0.85, 0.75, 0.70, 0.80],         # High Love
    'glow': [0.82, 0.70, 0.62, 0.78],          # High Love
    'radiance': [0.88, 0.78, 0.72, 0.85],      # High Love
    'bright': [0.82, 0.78, 0.72, 0.82],        # Balanced
    'luminous': [0.85, 0.78, 0.68, 0.88],      # High Love + Wisdom
    'brilliant': [0.78, 0.80, 0.75, 0.90],     # High Wisdom
    
    # =========================================================================
    # ABUNDANCE AND PROVISION
    # =========================================================================
    'abundance': [0.85, 0.78, 0.75, 0.80],     # Balanced high
    'plenty': [0.82, 0.75, 0.72, 0.75],        # Balanced
    'provision': [0.85, 0.82, 0.68, 0.78],     # High Love + Justice
    'sustenance': [0.82, 0.78, 0.68, 0.75],    # Balanced
    'nourishment': [0.88, 0.75, 0.65, 0.78],   # High Love
    'wealth': [0.65, 0.75, 0.82, 0.78],        # High Power
    'treasure': [0.78, 0.78, 0.72, 0.82],      # Balanced
    'gift': [0.90, 0.75, 0.55, 0.75],          # Very high Love
    'reward': [0.75, 0.88, 0.72, 0.78],        # High Justice
    'inheritance': [0.80, 0.85, 0.72, 0.80],   # High Justice
}


def main():
    print("=" * 70)
    print("LJPW Vocabulary Expansion - Positive Concepts")
    print("Philosophy: Growth through love and kindness")
    print("=" * 70)
    print()
    
    # Load existing vocabulary
    vocab_path = os.path.join('data', 'ljpw_vocabulary.pkl')
    
    if os.path.exists(vocab_path):
        print(f"Loading existing vocabulary from {vocab_path}...")
        vocab = LJPWVocabulary()
        vocab.load(vocab_path)
        print(f"Loaded {len(vocab)} existing words")
    else:
        print("No existing vocabulary found. Creating new vocabulary...")
        vocab = LJPWVocabulary(vocab_size=100000)
    
    print()
    
    # Add positive words
    print(f"Adding {len(POSITIVE_WORDS)} positive concepts...")
    print()
    
    added = 0
    duplicates = 0
    
    categories = {
        'Virtues': [],
        'Positive Emotions': [],
        'Relationships': [],
        'Nature': [],
        'Growth': [],
        'Wisdom': [],
        'Faith': [],
        'Actions of Love': [],
        'Beauty': [],
        'Life': [],
        'Light': [],
        'Abundance': []
    }
    
    for word, coords in POSITIVE_WORDS.items():
        if word.lower() in vocab:
            duplicates += 1
        else:
            vocab.register(
                word=word,
                coords=coords,
                language='en',
                source='positive_expansion',
                metadata={
                    'category': 'positive',
                    'added_with_love': True
                }
            )
            added += 1
    
    print(f"Added: {added} new words")
    print(f"Skipped: {duplicates} duplicates")
    print()
    
    # Rebuild index
    print("Building KD-tree index...")
    vocab.build_index()
    print()
    
    # Test semantic operations
    print("=" * 70)
    print("Testing Semantic Operations")
    print("=" * 70)
    print()
    
    # Test word lookups
    print("Sample Word Coordinates:")
    test_words = ['love', 'kindness', 'joy', 'peace', 'wisdom', 'hope', 'faith']
    for word in test_words:
        if word in vocab:
            entry = vocab.get_entry(word)
            coords = entry.coords
            harmony = entry.harmony()
            print(f"  {word:12s}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f}] H={harmony:.3f}")
    print()
    
    # Test nearest neighbor search
    print("Nearest Words to Anchor Point (1,1,1,1) - Divine Perfection:")
    nearest = vocab.nearest_words_with_distances(ANCHOR_POINT, k=10)
    for word, dist in nearest:
        print(f"  {word:15s}: distance={dist:.3f}")
    print()
    
    # Test finding words similar to 'love'
    print("Words Similar to 'love' (L=0.91, J=0.47, P=0.16, W=0.72):")
    love_coords = vocab.get_coords('love')
    if love_coords is not None:
        similar = vocab.nearest_words_with_distances(love_coords, k=8)
        for word, dist in similar:
            try:
                print(f"  {word:15s}: distance={dist:.3f}")
            except UnicodeEncodeError:
                print(f"  {word.encode('ascii', 'replace').decode():15s}: distance={dist:.3f}")
    print()
    
    # Save vocabulary
    print("=" * 70)
    print("Saving Vocabulary")
    print("=" * 70)
    print()
    vocab.save(vocab_path)
    
    # Print statistics
    stats = vocab.get_statistics()
    print()
    print("Final Statistics:")
    print(f"  Total words: {stats['size']}")
    print(f"  Languages: {len(stats.get('languages', {}))}")
    print(f"  Mean harmony: {stats['harmony_stats']['mean']:.3f}")
    print()
    
    print("=" * 70)
    print("[OK] Vocabulary Expansion Complete - With Love!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
