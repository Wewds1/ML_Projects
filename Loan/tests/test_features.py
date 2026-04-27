import pandas as pd

from src.loan_predictor.features import prepare_features


def test_prepare_features_adds_engineered_columns():
    df = pd.DataFrame(
        [
            {
                "borrower_id": "BRW1",
                "age": 30,
                "employment_type": "self employed ",
                "employment_years": 5.0,
                "annual_income": 80000.0,
                "credit_score": 710.0,
                "loan_amount": 20000.0,
                "loan_term_months": 36,
                "loan_purpose": "debt consolidation",
                "existing_debt": 5000.0,
                "num_open_accounts": 4,
                "num_late_payments": 1,
                "dti_ratio": 0.42,
                "origination_date": "2023-08-18",
                "origination_year": 2023,
                "origination_quarter": "Q3",
                "fed_funds_rate": 5.25,
                "interest_rate_offered": 9.1,
            }
        ]
    )

    result = prepare_features(df, training=True)

    assert result.loc[0, "employment_type"] == "Self-Employed"
    assert result.loc[0, "loan_purpose"] == "Debt Consolidation"
    assert "loan_to_income_ratio" in result.columns
    assert "payment_to_income" in result.columns
    assert "rate_spread" in result.columns
