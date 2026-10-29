"""Camera-source configuration and CLI for the shared downloader; offline by default."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import os

from dotenv import load_dotenv
from data.download import fetch_soda, checksum, save_json  # historical utility imports

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
CACHE = Path(os.getenv('CAMERA_TICKET_CACHE', ROOT / 'data/raw/camera_ticket_trends_20261014'))
START, END = '2016-01-01', '2026-08-31'
SOURCES = dict(zip(range(2017, 2028), [
    '2bnn-yakx', 'a5td-mswe', 'faiq-9dfq', 'p7t3-5i9s', 'kvfd-bves',
    '7mxj-7a6y', '869v-vr48', '8zf9-spf8', 'm5vz-tzqv', '9mwx-gamw', 'pvqr-7yc4',
]))
FIELDS = ['source_row_id', 'summons_number', 'issue_date', 'violation_code',
          'street_name', 'intersecting_street', 'violation_county', 'issuing_agency']
SELECT = ':id as source_row_id,' + ','.join(FIELDS[1:])
# The frozen official_violation_codes.json receipt documents these code selections.
BASE_WHERE = ('violation_code IN (7,36) AND '
              f"((issue_date >= '{START}' AND issue_date < '{END}') OR issue_date IS NULL)")


def fetch_source(fiscal_year, dataset, *, cache=CACHE, page_size=250000, offline=True):
    """Delegate all retrieval, caching and verification to the shared implementation."""
    return fetch_soda(dataset, f'camera_tickets_fy{fiscal_year}', BASE_WHERE,
                      columns=FIELDS, snapshot_dir=Path(cache) / f'fy{fiscal_year}',
                      partition_field='issue_date', fiscal_year=fiscal_year,
                      page_size=page_size, offline=offline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--download', action='store_true', help='Explicitly allow network retrieval of missing pages')
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fetch_source, fy, ds, cache=args.cache, offline=not args.download)
                for fy, ds in SOURCES.items()]
        counts = [job.result() for job in jobs]
    print(f'Verified {sum(counts):,} raw records across {len(counts)} fiscal-year sources.')


if __name__ == '__main__':
    main()
