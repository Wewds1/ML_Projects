# API Reference

## Endpoints

### `GET /`

Returns service metadata and documentation links.

### `GET /health`

Checks that the API is running and the model artifact is available.

### `POST /predict`

Scores one or more loan applications and returns predicted interest rates.

## Notes

- `annual_income` and `credit_score` may be omitted and will be imputed by the feature pipeline.
- `origination_date` is required.
- `origination_year` and `origination_quarter` can be omitted if derivable from `origination_date`.
- malformed payloads return `422`.
