"""
Extract instance subjects excluded with reason ``instance_no_label``.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

LOG_PATH = config.INST_TYPES_FOLDER / config.INST_META_MESSAGES_FILE
OUTPUT_PATH = config.NO_LABEL_INSTANCES_FILE

_INSTANCE_PROP = "wdt:P31"


def _subject_from_no_label_meta_line(line: str) -> str | None:
    """Return the instance subject when excluded for instance_no_label, else None."""
    parts = line.split("\t")
    if len(parts) < 7 or parts[0] != "<<" or parts[2] != _INSTANCE_PROP:
        return None
    if parts[6].strip() != "instance_no_label":
        return None
    return parts[1]


def extract_no_label_instances(
    log_path: str = LOG_PATH,
    output_path: str = OUTPUT_PATH):
    """
    Collect instance subjects excluded with reason ``instance_no_label``.

    Reads *log_path*, gathers unique subjects from matching ``wdt:P31`` meta
    lines, and writes one ``wd:Q…`` id per line (sorted) to *output_path*.

    Returns ``(count, no_label_instances)``.
    """
    no_label: set[str] = set()
    log_lines = 0
    print(f"Scanning for instance_no_label in {log_path} …")
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            log_lines += 1
            subject = _subject_from_no_label_meta_line(line)
            if subject is not None:
                no_label.add(subject)
            if log_lines % 5_000_000 == 0:
                print(
                    f"  … {log_lines:,} log lines, "
                    f"{len(no_label):,} unique subjects so far"
                )

    print(f"  done: {log_lines:,} log lines")
    print(f"\nWriting {len(no_label):,} subjects to {output_path} …")
    with open(output_path, "w", encoding="utf-8") as out:
        for inst in sorted(no_label):
            out.write(f"{inst}\n")

    print("\n" + "=" * 60)
    print(f"instance_no_label subjects (unique): {len(no_label):,}")
    print("=" * 60)

    return len(no_label)


if __name__ == "__main__":
    extract_no_label_instances()
