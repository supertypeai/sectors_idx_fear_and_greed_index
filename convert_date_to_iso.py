import locale
import pandas as pd

from argparse import ArgumentParser

from synchronize_data import (
    override_locale
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="the input file, in csv format"
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=["date"],
        nargs="+",
        help="columns containing human-readable date format that needs transformation",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/result.csv",
        help="output file",
    )

    args = parser.parse_args()
    file: str = args.file
    date_columns: list[str] = args.columns
    output: str = args.output

    with open(file) as f:
        with override_locale(locale.LC_TIME, 'id_ID.utf8'):
            df = pd.read_csv(f, parse_dates=date_columns, date_format='%d %B %Y')

    df.to_csv(output, index=False)
