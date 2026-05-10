# Risk Patterns

This part is where the data gets more intresting.

## Time stuff

Fraud counts peak around **hour 3**, but the fraud **rate** peaks around **hour 3**. That split matters a lot, otherwise we'd just be charting volume and calling it insight.

## Categorical risk

Top merchant categories by fraud rate:

| index | fraud_rate |
|---|---:|
| Healthcare | 0.0429 |
| ATM | 0.0405 |
| Luxury | 0.0375 |
| Online_Retail | 0.0319 |
| Gas | 0.0317 |

Top channels by fraud rate:

| index | fraud_rate |
|---|---:|
| Online | 0.0355 |
| ATM | 0.0267 |
| Chip | 0.0258 |
| Swipe | 0.0255 |
| Contactless | 0.0232 |

Top device types by fraud rate:

| index | fraud_rate |
|---|---:|
| Unknown | 0.0313 |
| POS | 0.0292 |
| Mobile | 0.0278 |
| Desktop | 0.0262 |
| ATM | 0.0237 |

The online-ish behavior and ATM-style behavior are defintely riskier here, which is pretty much what we hoped to confirm after the cleanup.
