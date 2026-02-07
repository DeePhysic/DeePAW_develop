#!/usr/bin/env python
"""
Command-line script to extract atomic embeddings from crystal structures.

Usage:
    python extract_script.py structure.cif --output embeddings.npz
    python extract_script.py structure.cif --output embeddings.npy --format npy
    python extract_script.py *.cif --output_dir ./embeddings/
"""

import argparse
import sys
from pathlib import Path
from extractor import AtomicEmbeddingExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Extract atomic embeddings from crystal structures'
    )

    # Input arguments
    parser.add_argument(
        'input',
        nargs='+',
        help='Input structure file(s) (CIF, POSCAR, xyz, etc.)'
    )

    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (for single input)'
    )

    parser.add_argument(
        '--output_dir', '-d',
        type=str,
        default='./embeddings',
        help='Output directory (for multiple inputs)'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['npz', 'npy', 'json'],
        default='npz',
        help='Output format (default: npz)'
    )

    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: use pretrained)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: auto-detect)'
    )

    # Processing arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed information'
    )

    args = parser.parse_args()

    # Initialize extractor
    print("Initializing embedding extractor...")
    extractor = AtomicEmbeddingExtractor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Process files
    input_files = [Path(f) for f in args.input]

    if len(input_files) == 1 and args.output:
        # Single file mode
        input_file = input_files[0]
        print(f"\nProcessing: {input_file}")

        embeddings = extractor.extract_from_file(input_file, return_numpy=True)

        if args.verbose:
            from ase.io import read
            atoms = read(str(input_file))
            stats = extractor.get_embedding_statistics(embeddings)
            print(f"  Atoms: {len(atoms)}")
            print(f"  Embeddings shape: {embeddings.shape}")
            print(f"  Statistics: {stats}")

        # Save
        from ase.io import read
        atoms = read(str(input_file))
        extractor.save_embeddings(embeddings, atoms, args.output, format=args.format)

    else:
        # Batch mode
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {len(input_files)} files...")
        print(f"Output directory: {output_dir}")

        for input_file in input_files:
            try:
                print(f"\n  Processing: {input_file.name}")

                embeddings = extractor.extract_from_file(input_file, return_numpy=True)

                if args.verbose:
                    from ase.io import read
                    atoms = read(str(input_file))
                    stats = extractor.get_embedding_statistics(embeddings)
                    print(f"    Atoms: {len(atoms)}")
                    print(f"    Embeddings shape: {embeddings.shape}")

                # Save with same name
                output_name = input_file.stem + f'_embeddings.{args.format}'
                output_path = output_dir / output_name

                from ase.io import read
                atoms = read(str(input_file))
                extractor.save_embeddings(embeddings, atoms, output_path, format=args.format)

            except Exception as e:
                print(f"    Error processing {input_file}: {e}")
                continue

        print(f"\nDone! Processed {len(input_files)} files.")


if __name__ == '__main__':
    main()
