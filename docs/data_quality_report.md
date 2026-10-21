# Data quality report

Executed against the original raw files; machine-readable evidence is in
[data_quality.json](../reports/reference/data_quality.json) and file hashes in
[sources.json](../reports/reference/sources.json).

- Input: **300,261** offers. Raw fields have no missing values.
- Removed **2** exact duplicate records.
- Quarantined **4** rows with malformed duration strings (`1.03h m`, `1.02h m`,
  `1.01h m`); no duration or fare was invented to repair them.
- Retained **300,255** offers. Invalid/nonpositive fares, invalid travel dates,
  airline/code mismatches and unsupported routes: **0** in this version.
- **283,737** retained rows repeat a canonical model feature vector; **11,111**
  canonical configurations have more than one distinct fare. They remain valid
  observations, grouped across splits rather than artificially averaged or removed.
- Clock/duration consistency flags **4** raw rows, coinciding with the malformed
  duration exclusions. There is no remaining mismatch in the validated version.
- Fares range from INR **1,105** to **123,071**; median **7,425**; 1st/99th percentiles
  **1,776 / 76,736**. Durations range from **50 to 2,990 minutes**; median **675**.
  These extremes are flags for interpretation, not automatic grounds for deletion.
- Rare carriers: StarAir **61**, Trujet **41** offers. Aggregate performance cannot
  establish reliability for these carriers. Carrier names describe the 2022 source.

Ingestion uses anchored parsers, fixed categorical domains, strict positive finite fares,
and explicit source schema checks. Its report contains overlapping exclusion reasons
and representative offending source rows. Missing fields or an entirely invalid dataset
fail explicitly. Full-source quality summaries establish validity; target EDA and all
model development use the training partition.

Splitting uses **3,491** connected components joining repeated airline/route/flight-ID
records across travel dates/cabins and identical canonical model inputs across flight
IDs. The largest component has **1,305** rows. Components are constructed without fares.
This protects these defined near-duplicates; it cannot identify every related multi-leg
offer because complete itinerary and scrape identifiers are absent.

Sampling bias remains substantial: six metro cities, uneven carriers/classes, a short
travel window and unknown sampling mechanism. Listed options are not a representative
sample of completed purchases. No claim about national fare levels or current supply
follows from the row count. Publisher timing and airline-count descriptions are not
fully consistent with the raw schema; both discrepancies are documented.
