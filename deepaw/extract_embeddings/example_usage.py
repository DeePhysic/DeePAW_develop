#!/usr/bin/env python
"""
Simple example demonstrating atomic embedding extraction.

This script shows how to:
1. Load a crystal structure
2. Extract atomic embeddings
3. Analyze and save the results
"""

import os
import sys
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

from deepaw.extract_embeddings import AtomicEmbeddingExtractor
from ase.build import bulk
from ase.io import read, write


def example_1_simple_extraction():
    """Example 1: Simple extraction from a built-in structure"""
    print("=" * 60)
    print("Example 1: Simple Extraction")
    print("=" * 60)

    # Initialize extractor
    print("\n1. Initializing extractor...")
    extractor = AtomicEmbeddingExtractor()

    # Create a simple structure
    print("\n2. Creating Si diamond structure...")
    atoms = bulk('Si', 'diamond', a=5.43)
    print(f"   Structure: {len(atoms)} atoms")
    print(f"   Formula: {atoms.get_chemical_formula()}")

    # Extract embeddings
    print("\n3. Extracting embeddings...")
    embeddings = extractor.extract(atoms)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Each atom has {embeddings.shape[1]} features")

    # Analyze embeddings
    print("\n4. Analyzing embeddings...")
    stats = extractor.get_embedding_statistics(embeddings)
    print(f"   Mean: {stats['mean']:.4f}")
    print(f"   Std: {stats['std']:.4f}")
    print(f"   Norm (mean): {stats['norm_mean']:.4f}")

    # Save embeddings
    print("\n5. Saving embeddings...")
    output_dir = os.path.join(deepaw_root, 'outputs', 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'si_embeddings.npz')
    extractor.save_embeddings(embeddings, atoms, output_path)

    print("\n✓ Example 1 completed!")
    return embeddings


def example_2_from_file():
    """Example 2: Extract from a file"""
    print("\n" + "=" * 60)
    print("Example 2: Extract from File")
    print("=" * 60)

    # Check if example file exists
    example_file = os.path.join(deepaw_root, 'examples', 'hfo2_chgd.db')
    if not os.path.exists(example_file):
        print(f"\n⚠ Example file not found: {example_file}")
        print("   Skipping this example.")
        return None

    # Initialize extractor
    print("\n1. Initializing extractor...")
    extractor = AtomicEmbeddingExtractor()

    # Load structure from database
    print("\n2. Loading structure from database...")
    from ase.db import connect
    db = connect(example_file)
    row = db.get(1)
    atoms = row.toatoms()
    print(f"   Structure: {len(atoms)} atoms")
    print(f"   Formula: {atoms.get_chemical_formula()}")

    # Extract embeddings
    print("\n3. Extracting embeddings...")
    embeddings = extractor.extract(atoms)
    print(f"   Embeddings shape: {embeddings.shape}")

    # Analyze per-element embeddings
    print("\n4. Analyzing per-element embeddings...")
    symbols = atoms.get_chemical_symbols()
    unique_symbols = set(symbols)

    for symbol in unique_symbols:
        indices = [i for i, s in enumerate(symbols) if s == symbol]
        element_embeddings = embeddings[indices]
        norm_mean = np.linalg.norm(element_embeddings, axis=1).mean()
        print(f"   {symbol}: {len(indices)} atoms, norm={norm_mean:.4f}")

    print("\n✓ Example 2 completed!")
    return embeddings


def example_3_batch_processing():
    """Example 3: Batch processing multiple structures"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    # Initialize extractor
    print("\n1. Initializing extractor...")
    extractor = AtomicEmbeddingExtractor()

    # Create multiple structures
    print("\n2. Creating multiple structures...")
    structures = [
        bulk('Si', 'diamond', a=5.43),
        bulk('C', 'diamond', a=3.57),
        bulk('Ge', 'diamond', a=5.66),
    ]
    print(f"   Created {len(structures)} structures")

    # Batch extraction
    print("\n3. Batch extracting embeddings...")
    embeddings_list = extractor.extract_batch(structures, show_progress=True)

    # Analyze results
    print("\n4. Analyzing results...")
    for i, (atoms, embeddings) in enumerate(zip(structures, embeddings_list)):
        formula = atoms.get_chemical_formula()
        stats = extractor.get_embedding_statistics(embeddings)
        print(f"   Structure {i} ({formula}):")
        print(f"      Shape: {embeddings.shape}")
        print(f"      Norm: {stats['norm_mean']:.4f}")

    print("\n✓ Example 3 completed!")
    return embeddings_list


def example_4_similarity_analysis():
    """Example 4: Similarity analysis between structures"""
    print("\n" + "=" * 60)
    print("Example 4: Similarity Analysis")
    print("=" * 60)

    # Initialize extractor
    print("\n1. Initializing extractor...")
    extractor = AtomicEmbeddingExtractor()

    # Create two similar structures
    print("\n2. Creating structures...")
    atoms1 = bulk('Si', 'diamond', a=5.43)
    atoms2 = bulk('Si', 'diamond', a=5.50)  # Slightly different lattice
    print(f"   Structure 1: a={atoms1.cell[0, 0]:.2f} Å")
    print(f"   Structure 2: a={atoms2.cell[0, 0]:.2f} Å")

    # Extract embeddings
    print("\n3. Extracting embeddings...")
    emb1 = extractor.extract(atoms1)
    emb2 = extractor.extract(atoms2)

    # Compute similarity
    print("\n4. Computing similarity...")
    from sklearn.metrics.pairwise import cosine_similarity

    # Structure-level similarity (average embeddings)
    struct_emb1 = emb1.mean(axis=0, keepdims=True)
    struct_emb2 = emb2.mean(axis=0, keepdims=True)
    similarity = cosine_similarity(struct_emb1, struct_emb2)[0, 0]

    print(f"   Structure similarity: {similarity:.6f}")
    print(f"   (1.0 = identical, 0.0 = orthogonal)")

    print("\n✓ Example 4 completed!")
    return similarity


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Atomic Embedding Extraction Examples")
    print("=" * 60)

    try:
        # Run examples
        example_1_simple_extraction()
        example_2_from_file()
        example_3_batch_processing()
        example_4_similarity_analysis()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
