# Data dictionary and provenance

Original source: Shubham Bathwal, [Flight Price Prediction on Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).
The public metadata endpoint reports version 2, publisher license **CC0: Public Domain**,
and EaseMyTrip/Octoparse as the extraction source/method. Source metadata was retrieved
on 21 October 2026; see [recorded provenance](../reports/reference/provenance.json).
The repository's MIT license covers project code; the dataset publisher's terms are
separate. No account credentials are stored or required by the anonymous downloader.

The publisher describes a February-March 2022 collection window and calls it 50 days.
Actual raw travel dates span 11 February through 31 March (49 distinct calendar dates).
A date on an offer is a travel date; the publisher narrative does not establish a
per-row scrape/booking timestamp. Raw inspection takes precedence over inferred timing.

Run `python scripts/download_data.py` to retrieve pinned Kaggle version 2. If anonymous
downloads fail, configure the official Kaggle CLI yourself and run
`kaggle datasets download -d shubhambathwal/flight-price-prediction -p data/raw --unzip`.
Verify the resulting file hashes against [sources](../reports/reference/sources.json).
Keep CSVs out of version control. The downloader records metadata; ingestion records
SHA-256, row counts and schema checks.

| Raw field | Meaning / type | Availability and treatment |
|---|---|---|
| `date` | Travel date, DD-MM-YYYY | Known travel detail; validated metadata; excluded as a predictor. |
| `airline` | Historical carrier name | Known; categorical predictor. Eight names in raw data, despite six in the cleaned-source description. |
| `ch_code` | Carrier prefix | Known; checked against airline; excluded as redundant. |
| `num_code` | Flight numeric identifier, stored as string | Used only for repeated-itinerary grouping. Not a predictor. |
| `dep_time`, `arr_time` | Scheduled HH:MM | Known; mapped to six deterministic time buckets. |
| `from`, `to` | Origin/destination city | Known; directional categorical predictors, six supported cities. |
| `time_taken` | Total itinerary duration, hours/minutes | Known; parsed into minutes, including connections and multi-day duration. |
| `stop` | Non-stop, one stop, two-plus stops | Known; mapped to 0/1/2, where 2 means two or more. |
| `price` | Listed fare, positive INR | Target only. Commas stripped before numeric validation. |
| filename | Economy or business source | Cabin label supplied by source-file identity. |

Canonical schema v2 contains six categorical columns: `airline`, `source_city`,
`destination_city`, `cabin_class`, `departure_time`, `arrival_time`; and two numeric
columns: `duration_minutes`, `stops`. The canonical request contains neither price,
flight ID, travel date nor booking lead time.

Files: economy.csv (206,774 rows), business.csv (93,487 rows), total 300,261 raw offers.
These counts are verified by ingestion. `Clean_Dataset.csv` is an alternative processed
table; **it is not concatenated with the raw files**. Its different counts, rounded
durations, carrier filtering and derived `days_left` make it unsuitable as an
interchangeable training source without a separately audited adapter.

Missing for operational use: observation timestamps, itinerary/offer identifiers across
scrapes, inventory, taxes/inclusions, refundability, baggage, airport and connection
details, marketing/operating carrier distinctions, demand, purchase/conversion outcomes,
and evidence that any offer was actually booked.
