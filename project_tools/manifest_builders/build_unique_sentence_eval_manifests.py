from collections import defaultdict
from pathlib import Path
import csv

ROOT = Path("/data/Database/RESPIN/FINAL_DATA")
OUT = Path("/data/Kunal/manifests")
LANGS = ("bh", "bn", "ch", "kn", "mg", "mr", "mt", "te")
SPLITS = ("dev", "test")


def read_map(directory: Path, filename: str) -> dict[str, str]:
    path = directory / filename
    result: dict[str, str] = {}

    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line in {path}:{line_number}: {line!r}"
                )

            key, value = parts
            result[key] = value

    return result


def parse_gender(utt_id: str) -> str:
    for token in utt_id.split("_"):
        if token in {"M", "F"}:
            return token

    return "NA"


def build_split(split: str) -> tuple[Path, list[dict[str, object]]]:
    output_rows: list[dict[str, str | int]] = []
    summaries: list[dict[str, object]] = []

    for lang in LANGS:
        directory = (
            ROOT
            / "kaldi_files"
            / lang
            / f"IISc_RESPIN_{split}_{lang}"
        )

        wav = read_map(directory, "wav.scp")
        text = read_map(directory, "text")
        dialect = read_map(directory, "utt2dialect")
        speaker = read_map(directory, "utt2spk")
        duration = read_map(directory, "utt2dur")
        language = read_map(directory, "utt2lang")

        mappings = {
            "text": text,
            "utt2dialect": dialect,
            "utt2spk": speaker,
            "utt2dur": duration,
            "utt2lang": language,
        }

        reference_keys = set(wav)

        for name, mapping in mappings.items():
            keys = set(mapping)

            if keys != reference_keys:
                raise RuntimeError(
                    f"{split}/{lang}: key mismatch in {name}; "
                    f"missing={len(reference_keys - keys)}, "
                    f"extra={len(keys - reference_keys)}"
                )

        by_text: dict[str, list[str]] = defaultdict(list)

        for utt_id in sorted(wav):
            by_text[text[utt_id]].append(utt_id)

        with_gender = 0
        without_gender = 0

        for transcript in sorted(by_text):
            candidates = by_text[transcript]

            # Prefer a recording with an explicit M/F gender label.
            labelled = [
                utt_id
                for utt_id in candidates
                if parse_gender(utt_id) in {"M", "F"}
            ]

            if labelled:
                utt_id = labelled[0]
                gender = parse_gender(utt_id)
                gender_available = 1
                with_gender += 1
            else:
                utt_id = candidates[0]
                gender = "NA"
                gender_available = 0
                without_gender += 1

            audio_path = Path(wav[utt_id])

            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Missing audio for {utt_id}: {audio_path}"
                )

            if language[utt_id] != lang:
                raise RuntimeError(
                    f"Language mismatch for {utt_id}: "
                    f"directory={lang}, label={language[utt_id]}"
                )

            output_rows.append(
                {
                    "utt_id": utt_id,
                    "audio_path": str(audio_path),
                    "split": split,
                    "language": lang,
                    "dialect": f"{lang}_{dialect[utt_id]}",
                    "dialect_local": dialect[utt_id],
                    "gender": gender,
                    "gender_available": gender_available,
                    "speaker_id": speaker[utt_id],
                    "duration": duration[utt_id],
                    "text": transcript,
                }
            )

        summaries.append(
            {
                "language": lang,
                "original_utterances": len(wav),
                "unique_sentences": len(by_text),
                "with_gender": with_gender,
                "without_gender": without_gender,
            }
        )

    output_path = OUT / f"respin_unique_sentence_{split}.csv"

    fieldnames = [
        "utt_id",
        "audio_path",
        "split",
        "language",
        "dialect",
        "dialect_local",
        "gender",
        "gender_available",
        "speaker_id",
        "duration",
        "text",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return output_path, summaries


OUT.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    manifest, summaries = build_split(split)

    print(f"\n===== {split.upper()} UNIQUE-SENTENCE MANIFEST =====")

    total_original = 0
    total_unique = 0
    total_with_gender = 0
    total_without_gender = 0

    for item in summaries:
        total_original += int(item["original_utterances"])
        total_unique += int(item["unique_sentences"])
        total_with_gender += int(item["with_gender"])
        total_without_gender += int(item["without_gender"])

        print(
            f"{item['language']}: "
            f"original={item['original_utterances']:,}  "
            f"unique={item['unique_sentences']:,}  "
            f"gender-labelled={item['with_gender']:,}  "
            f"gender-NA={item['without_gender']:,}"
        )

    print(
        f"total: original={total_original:,}  "
        f"unique={total_unique:,}  "
        f"gender-labelled={total_with_gender:,}  "
        f"gender-NA={total_without_gender:,}"
    )
    print(f"saved: {manifest}")
