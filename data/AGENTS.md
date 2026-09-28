# data/: raw data layer

**Job:** download raw data into the local database, record where it came from, and return it as DataFrames through `reader.py`. Nothing else happens here.

## Rules
- **Never change values.** Don't fill NAs (missing values are often meaningful, e.g. missing crash coordinates), don't create dummy variables, don't recode, and don't deduplicate on judgment calls.
- **Record provenance with every load:** dataset ID, the full query or filter, and the pull timestamp.
- Limiting **which records** you collect (a date range, violation codes) is allowed only as an explicit, logged pull parameter. Filtering on modeling criteria (e.g. fatal crashes only) belongs in `preprocessing/`.
- Keep all database access inside this folder so the backend can be swapped later (to Postgres or GCP).

## Downloading from NYC Open Data
- **Whole datasets:** use the bulk export at `https://data.cityofnewyork.us/api/views/<id>/rows.csv?accessType=DOWNLOAD`.
- **Filtered pulls:** use the SODA API at `https://data.cityofnewyork.us/resource/<id>.json` with `$where` and page with `$limit=50000&$offset=…&$order=:id`. Send the app token as the `X-App-Token` header, read from `.env`.
- **Parking violations:** there's one dataset per fiscal year, and each is very large (likely tens of millions of rows). **Never pull a full year.** Filter to camera violation codes with `$where=violation_code in (...)`.
- **Violation codes:** look them up in `ncbg-6agr`. Never hard-code them from memory.
- Store data as Parquet, GeoParquet or GeoPackage, or in the local database. Never commit data files.
