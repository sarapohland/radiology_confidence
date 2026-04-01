"""
Inspect the RadLex lexicon produced by load_radlex_lexicon().

Prints words in alphabetical order with basic statistics.
Supports optional prefix filtering and length filtering.

Usage
-----
    # Show all words
    python scripts/inspect_lexicon.py --lexicon_path lexicon/RadLex.owl

    # Show only words starting with 'pneu'
    python scripts/inspect_lexicon.py --lexicon_path lexicon/RadLex.owl --prefix pneu

    # Show only words of a specific length
    python scripts/inspect_lexicon.py --lexicon_path lexicon/RadLex.owl --length 8

    # Show words sorted by length (longest first)
    python scripts/inspect_lexicon.py --lexicon_path lexicon/RadLex.owl --sort length
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from logits import load_radlex_lexicon


def print_columns(words: list, ncols: int = 4, col_width: int = 22):
    for i in range(0, len(words), ncols):
        row = words[i:i + ncols]
        print("  " + "  ".join(w.ljust(col_width) for w in row))


def main():
    parser = argparse.ArgumentParser(description="Inspect the RadLex lexicon word set.")
    parser.add_argument("--lexicon_path", default="lexicon/RadLex.owl",
                        help="Path to RadLex.owl (default: lexicon/RadLex.owl).")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Filter to words starting with this prefix.")
    parser.add_argument("--length", type=int, default=None,
                        help="Filter to words of exactly this length.")
    parser.add_argument("--sort", choices=["alpha", "length"], default="alpha",
                        help="Sort order: alpha (default) or length (longest first).")
    args = parser.parse_args()

    print(f"Loading {args.lexicon_path} ...")
    words = load_radlex_lexicon(args.lexicon_path)
    print(f"Total unique terms: {len(words):,}\n")

    # Length distribution
    from collections import Counter
    length_dist = Counter(len(w) for w in words)
    print("Length distribution:")
    for length in sorted(length_dist):
        bar = "█" * (length_dist[length] // 20)
        print(f"  {length:3d}  {length_dist[length]:5d}  {bar}")
    print()

    # Apply filters
    filtered = sorted(words)

    if args.prefix:
        filtered = [w for w in filtered if w.startswith(args.prefix.lower())]
        print(f"Words starting with '{args.prefix}': {len(filtered)}\n")

    if args.length:
        filtered = [w for w in filtered if len(w) == args.length]
        print(f"Words of length {args.length}: {len(filtered)}\n")

    if args.sort == "length":
        filtered = sorted(filtered, key=lambda w: (-len(w), w))

    print_columns(filtered)
    print(f"\n({len(filtered)} words shown)")


if __name__ == "__main__":
    main()
