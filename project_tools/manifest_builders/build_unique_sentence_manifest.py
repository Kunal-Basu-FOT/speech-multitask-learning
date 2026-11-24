from collections import defaultdict
from pathlib import Path
import csv

ROOT = Path("/data/Database/RESPIN/FINAL_DATA")
OUT = Path("/data/Kunal/manifests")
LANGS = ("bh", "bn", "ch", "kn", "mg", "mr", "mt", "te")


def read_map(directory: Path, filename: str) -> dict[str, str]:
    result = {}

    with (directory / filename).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed {filename} line {line_number}: {line!r}"
                )

            result[parts[0]] = parts[1]

    return result


def parse_gender(utt_id: str) -> str:
    tokens = utt_id.split("_")

    for token in tokens:
        if token in {"M", "F"}:
            return token

    return "NA"


OUT.mkdir(parents=True, exist_ok=True)

rows = []
summary = []

for lang in LANGS:
    directory = (
        ROOT
        / "kaldi_files"
        / lang
        / f"IISc_RESPIN_train_{lang}_small"
    )

    wav = read_map(directory, "wav.scp")
    text = read_map(directory, "text")
    dialect = read_map(directory, "utt2dialect")
    speaker = read_map(directory, "utt2spk")
    duration = read_map(directory, "utt2dur")

    reference = set(wav)

    for name, mapping in {
        "text": text,
        "utt2dialect": dialect,
        "utt2spk": speaker,
        "utt2dur": duration,
    }.items():
        keys = set(mapping)

        if keys != reference:
            raise RuntimeError(
                f"{lang}: key mismatch in {name}; "
                f"missing={len(reference - keys)}, "
                f"extra={len(keys - reference)}"
            )

    # Group all recordings by their transcription.
    by_text = defaultdict(list)

    for utt_id in sorted(wav):
        by_text[text[utt_id]].append(utt_id)

    kept_with_gender = 0
    kept_without_gender = 0

    for transcript in sorted(by_text):
        candidates = by_text[transcript]

        # Prefer a gender-labelled recording.
        labelled = [
            utt_id
            for utt_id in candidates
            if parse_gender(utt_id) in {"M", "F"}
        ]

        if labelled:
            utt_id = labelled[0]
            gender = parse_gender(utt_id)
            gender_available = 1
            kept_with_gender += 1
        else:
            utt_id = candidates[0]
            gender = "NA"
            gender_available = 0
            kept_without_gender += 1

        audio_path = Path(wav[utt_id])

        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio: {audio_path}")

        rows.append(
            {
                "utt_id": utt_id,
                "audio_path": str(audio_path),
                "split": "train",
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

    summary.append(
        {
            "language": lang,
            "unique_sentences": len(by_text),
            "with_gender": kept_with_gender,
            "without_gender": kept_without_gender,
        }
    )

manifest = OUT / "respin_unique_sentence_train.csv"

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

with manifest.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("===== UNIQUE-SENTENCE MANIFEST =====")

for item in summary:
    print(
        f"{item['language']}: "
        f"unique={item['unique_sentences']:,}  "
        f"gender-labelled={item['with_gender']:,}  "
        f"gender-NA={item['without_gender']:,}"
    )

print()
print(f"total rows: {len(rows):,}")
print(
    "gender-labelled rows:",
    f"{sum(x['with_gender'] for x in summary):,}",
)
print(
    "gender-NA rows:",
    f"{sum(x['without_gender'] for x in summary):,}",
)
print(f"saved: {manifest}")
