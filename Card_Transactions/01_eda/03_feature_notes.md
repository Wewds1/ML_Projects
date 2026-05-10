# Feature Notes

These are the engineered bits that probly matter most before modeling:

- `amount_to_limit_ratio`: catches pressure against the credit limit, not just raw spend.
- `amount_vs_7day_avg`: flags when a charge is way above the cardholder's usual rhythm.
- `txn_velocity_ratio`: tells us if the last 24 hours are moving way faster than the last week.
- `foreign_online`, `late_night_foreign`, `new_account_high_amount`: these combo flags are kinda where the fraud story gets more real.

## Correlation check

Top numeric features by point-biserial correlation with fraud:

| index | correlation |
|---|---:|
| declined_last_30d | 0.0550 |
| is_foreign | 0.0523 |
| foreign_online | 0.0469 |
| is_late_night | 0.0366 |
| late_night_foreign | 0.0321 |
| txn_velocity_ratio | 0.0214 |
| merchant_country_missing | 0.0203 |
| is_new_account | 0.0145 |
| days_since_last_missing | 0.0136 |
| days_since_last_txn | 0.0135 |

It's not a perfect ranking, but it gives a good sanity check before fitting the heavier models.
