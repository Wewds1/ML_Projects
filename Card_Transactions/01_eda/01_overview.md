# EDA Overview

This dataset starts at **9,025 rows** and lands at **9,000 rows** after the duplicate cleanup, so yeah, the batch issue in the brief was real.

The fraud rate is **2.80%**. That number kinda rules the whole project, because accuracy would look fake-good here and still miss every bad transaction.

Quick notes from the first pass:

- Duplicate rows found when ignoring `transaction_id`: **25**
- Missing `merchant_country`: **361**
- Missing `avg_7day_spend`: **271**
- Missing `days_since_last_txn`: **270**

The fraud class is tiny, but it does not look random. A couple patterns show up pretty fast once the dirty categories get fixed.

## Files saved

- `figures/fraud_count_by_hour.png`
- `figures/fraud_rate_by_hour.png`
- `figures/fraud_rate_by_day.png`
- `figures/missingness_matrix.png`
- `point_biserial_correlations.csv`
