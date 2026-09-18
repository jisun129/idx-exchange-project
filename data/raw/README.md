# data/raw/

Raw CRMLS (California Regional MLS) sold-listing CSV files go here.

**These files are not included in this repository.** Under this project's data
non-disclosure policy, CRMLS listing data cannot be redistributed publicly, so `data/raw/` is
excluded via `.gitignore` (see the main [README](../README.md), Section 2, for detail).

## To reproduce this project

1. Obtain the raw CRMLS sold-listing data yourself (through whatever access you have to
   CRMLS).
2. Place the monthly files here, named `CRMLSSold{YYYYMM}.csv` (e.g. `CRMLSSold202505.csv`).
3. Run `src/preprocess.py` (or the notebooks in `notebooks/`) as documented in the main
   README.

By default, only files matching `CRMLSSold{YYYYMM}.csv` for `YYYYMM` in the range
`202505`-`202605` are used - see the main README (Sections 3 and 6.1) for detail, or pass
`--start`/`--end` to `src/preprocess.py` to use a different range.
