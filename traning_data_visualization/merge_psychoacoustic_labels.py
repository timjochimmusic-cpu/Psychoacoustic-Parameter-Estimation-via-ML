"""
Merge all psychoacoustic label CSV files from a given directory
into a single CSV file.
"""

import csv
import glob
import os


CSV_FIELDS = [
    "time_index",
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]

OUTPUT_FIELDS = ["source_file"] + CSV_FIELDS


def merge_psychoacoustic_labels(labels_dir):
    """
    Merge all psychoacoustic label CSV files in a directory.

    Parameters
    ----------
    labels_dir : str
        Path to the directory containing the psychoacoustic label CSV files.
    """
    output_dir = os.path.join(
        os.path.dirname(labels_dir),
        "merged_labels",
    )
    output_file = os.path.join(
        output_dir,
        "all_psychoacoustic_labels.csv",
    )

    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(labels_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    total = len(files)

    print(f"Found {total} CSV files to merge")

    written = 0

    with open(output_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()

        for i, fpath in enumerate(files, 1):
            fname = os.path.basename(fpath)

            with open(fpath, newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)

                for row in reader:
                    out_row = {"source_file": fname}

                    for field in CSV_FIELDS:
                        val = row.get(field, "")

                        if val is None:
                            val = ""

                        out_row[field] = val

                    writer.writerow(out_row)
                    written += 1

            if i % 5000 == 0:
                print(
                    f"Processed {i}/{total} files "
                    f"({written} rows written)"
                )

    print(f"\nDone! Merged {total} files into {output_file}")
    print(f"Total rows: {written}")