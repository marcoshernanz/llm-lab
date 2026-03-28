import argparse
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Sequence

BYTE_VOCAB_SIZE = 256
DEFAULT_SPLIT_PATTERN = r"\s+\S+|\S+|\s+"
TOKENIZER_ARTIFACT_VERSION = 1

type TokenId = int
type TokenPair = tuple[TokenId, TokenId]
type TokenSequence = list[TokenId]
type TokenKey = tuple[TokenId, ...]
type Merge = tuple[TokenPair, TokenId]


@dataclass(frozen=True, slots=True)
class BPEModel:
    split_pattern: str
    vocab: dict[TokenId, bytes]
    merges: tuple[Merge, ...]
    merge_ranks: dict[TokenPair, int]
    merge_tokens: dict[TokenPair, TokenId]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[TokenId]:
        token_ids: list[TokenId] = []
        for chunk in split_text(text, self.split_pattern):
            token_ids.extend(self.encode_chunk(chunk))
        return token_ids

    def encode_chunk(self, chunk: bytes) -> list[TokenId]:
        sequence = list(chunk)
        while True:
            pair = select_best_mergeable_pair(sequence, self.merge_ranks)
            if pair is None:
                return sequence
            sequence = merge_sequence(sequence, pair, self.merge_tokens[pair])

    def decode(self, token_ids: Sequence[TokenId]) -> str:
        decoded = b"".join(self.vocab[token_id] for token_id in token_ids)
        return decoded.decode("utf-8")

    def decode_for_display(self, token_ids: Sequence[TokenId]) -> str:
        decoded = b"".join(self.vocab[token_id] for token_id in token_ids)
        return decoded.decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": TOKENIZER_ARTIFACT_VERSION,
            "split_pattern": self.split_pattern,
            "merge_pairs": [list(pair) for pair, _ in self.merges],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BPEModel":
        version = payload.get("version")
        if version != TOKENIZER_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported tokenizer artifact version {version!r}; "
                f"expected {TOKENIZER_ARTIFACT_VERSION}."
            )

        split_pattern = payload.get("split_pattern")
        if not isinstance(split_pattern, str):
            raise ValueError("Tokenizer artifact must contain a string split_pattern.")

        merge_pairs_payload = payload.get("merge_pairs")
        if not isinstance(merge_pairs_payload, list):
            raise ValueError("Tokenizer artifact must contain a list merge_pairs.")

        merges = build_merges_from_pairs(merge_pairs_payload)
        return build_model(split_pattern, merges)

    @classmethod
    def load(cls, path: Path) -> "BPEModel":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def split_text(text: str, split_pattern: str = DEFAULT_SPLIT_PATTERN) -> list[bytes]:
    return [chunk.encode("utf-8") for chunk in re.findall(split_pattern, text)]


def count_pairs(sequence_counts: dict[TokenKey, int]) -> dict[TokenPair, int]:
    pair_counts: dict[TokenPair, int] = {}
    for sequence, frequency in sequence_counts.items():
        for pair, count in count_sequence_pairs(sequence).items():
            pair_counts[pair] = pair_counts.get(pair, 0) + (count * frequency)
    return pair_counts


def count_sequence_pairs(sequence: Sequence[TokenId]) -> Counter[TokenPair]:
    return Counter(zip(sequence, sequence[1:]))


def select_best_pair(pair_counts: dict[TokenPair, int]) -> TokenPair | None:
    if not pair_counts:
        return None
    return min(pair_counts.items(), key=lambda item: (-item[1], item[0]))[0]


def select_best_mergeable_pair(
    sequence: Sequence[TokenId],
    merge_ranks: dict[TokenPair, int],
) -> TokenPair | None:
    best_pair: TokenPair | None = None
    best_rank: int | None = None
    for pair in zip(sequence, sequence[1:]):
        rank = merge_ranks.get(pair)
        if rank is None:
            continue
        if best_rank is None or rank < best_rank:
            best_pair = pair
            best_rank = rank
    return best_pair


def merge_sequence(
    sequence: Sequence[TokenId], pair: TokenPair, new_token_id: TokenId
) -> TokenSequence:
    merged: TokenSequence = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


def build_vocab(merges: Sequence[Merge]) -> dict[TokenId, bytes]:
    vocab = {token_id: bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)}
    for pair, token_id in merges:
        vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]
    return vocab


def build_merge_ranks(merges: Sequence[Merge]) -> dict[TokenPair, int]:
    return {pair: rank for rank, (pair, _) in enumerate(merges)}


def build_merge_tokens(merges: Sequence[Merge]) -> dict[TokenPair, TokenId]:
    return {pair: token_id for pair, token_id in merges}


def build_merges_from_pairs(merge_pairs: Sequence[object]) -> tuple[Merge, ...]:
    merges: list[Merge] = []
    next_token_id = BYTE_VOCAB_SIZE
    for merge_pair in merge_pairs:
        if not isinstance(merge_pair, list) or len(merge_pair) != 2:
            raise ValueError("Each merge pair must be a two-item list of token ids.")
        left_token_id, right_token_id = merge_pair
        if not isinstance(left_token_id, int) or not isinstance(right_token_id, int):
            raise ValueError("Merge pair token ids must be integers.")
        merges.append(((left_token_id, right_token_id), next_token_id))
        next_token_id += 1
    return tuple(merges)


def build_model(split_pattern: str, merges: Sequence[Merge]) -> BPEModel:
    frozen_merges = tuple(merges)
    return BPEModel(
        split_pattern=split_pattern,
        vocab=build_vocab(frozen_merges),
        merges=frozen_merges,
        merge_ranks=build_merge_ranks(frozen_merges),
        merge_tokens=build_merge_tokens(frozen_merges),
    )


def train_bpe(
    text: str,
    vocab_size: int,
    *,
    split_pattern: str = DEFAULT_SPLIT_PATTERN,
) -> BPEModel:
    if vocab_size < BYTE_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be at least {BYTE_VOCAB_SIZE} for byte-level BPE.")

    sequence_counts = Counter(tuple(chunk) for chunk in split_text(text, split_pattern))
    pair_counts: Counter[TokenPair] = Counter()
    pair_to_sequences: dict[TokenPair, set[TokenKey]] = defaultdict(set)

    for sequence, frequency in sequence_counts.items():
        for pair, count in count_sequence_pairs(sequence).items():
            pair_counts[pair] += count * frequency
            pair_to_sequences[pair].add(sequence)

    merges: list[Merge] = []
    next_token_id = BYTE_VOCAB_SIZE

    while next_token_id < vocab_size:
        best_pair = select_best_pair(pair_counts)
        if best_pair is None:
            break

        affected_sequences = tuple(pair_to_sequences.get(best_pair, ()))
        if not affected_sequences:
            del pair_counts[best_pair]
            continue

        merged_sequence_counts: Counter[TokenKey] = Counter()
        for sequence in affected_sequences:
            frequency = sequence_counts.pop(sequence, 0)
            if frequency == 0:
                continue

            for pair, count in count_sequence_pairs(sequence).items():
                updated_count = pair_counts[pair] - (count * frequency)
                if updated_count > 0:
                    pair_counts[pair] = updated_count
                else:
                    del pair_counts[pair]

                indexed_sequences = pair_to_sequences[pair]
                indexed_sequences.discard(sequence)
                if not indexed_sequences:
                    del pair_to_sequences[pair]

            merged_sequence = tuple(merge_sequence(sequence, best_pair, next_token_id))
            merged_sequence_counts[merged_sequence] += frequency

        for sequence, frequency in merged_sequence_counts.items():
            sequence_counts[sequence] += frequency
            for pair, count in count_sequence_pairs(sequence).items():
                pair_counts[pair] += count * frequency
                pair_to_sequences[pair].add(sequence)

        merges.append((best_pair, next_token_id))
        print(f"merges_completed={next_token_id - BYTE_VOCAB_SIZE + 1}")
        next_token_id += 1

    return build_model(split_pattern, merges)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal byte-level BPE tokenizer.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the UTF-8 training corpus.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Final vocabulary size, including the 256 byte tokens.",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=None,
        help="Optional character limit for quick experiments.",
    )
    parser.add_argument(
        "--split-pattern",
        default=DEFAULT_SPLIT_PATTERN,
        help="Regex used to split text into independently merged chunks.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the tokenizer JSON artifact to write.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    text = args.data_path.read_text(encoding="utf-8")
    if args.text_limit is not None:
        text = text[: args.text_limit]

    model = train_bpe(
        text,
        args.vocab_size,
        split_pattern=args.split_pattern,
    )
    model.save(args.output_path)
    print(f"trained {len(model.merges)} merges")
    print(f"vocab size: {model.vocab_size}")
    print(f"saved tokenizer to {args.output_path}")


if __name__ == "__main__":
    main()
