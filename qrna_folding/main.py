import sys
from typing import List


def is_rna_valid(sequence: str) -> None:
    valid_chars = set("AUGC")
    if not all(base in valid_chars for base in sequence):
        raise ValueError(
            f"RNA sequence {sequence} is invalid. Only 'A', 'U', 'G', and 'C' characters are allowed."
        )
    return


def main(argv: List[str]) -> None:
    rna_sequence = input("Enter RNA sequence: ")
    is_rna_valid(rna_sequence)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
