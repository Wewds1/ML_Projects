const form = document.getElementById("prediction-form");
const resultShell = document.getElementById("result-shell");
const emptyState = document.getElementById("empty-state");
const resultChip = document.getElementById("result-chip");
const actionText = document.getElementById("action-text");
const probabilityText = document.getElementById("probability-text");
const thresholdText = document.getElementById("threshold-text");
const modelText = document.getElementById("model-text");

function setLoadingState() {
  emptyState.classList.add("hidden");
  resultShell.classList.remove("hidden");
  resultChip.className = "result-chip loading";
  resultChip.textContent = "SCORING";
  actionText.textContent = "Running model...";
  probabilityText.textContent = "--";
  thresholdText.textContent = "--";
  modelText.textContent = "--";
}

function castFormData(data) {
  const numericFields = new Set([
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "cardholder_age",
    "account_age_days",
    "credit_limit",
    "is_foreign",
    "transaction_amount",
    "avg_7day_spend",
    "num_txn_24h",
    "num_txn_7d",
    "days_since_last_txn",
    "distinct_merchants_7d",
    "declined_last_30d"
  ]);

  const payload = {};
  for (const [key, value] of data.entries()) {
    if (value === "") {
      payload[key] = null;
    } else if (numericFields.has(key)) {
      payload[key] = Number(value);
    } else {
      payload[key] = value;
    }
  }
  return payload;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoadingState();

  const payload = castFormData(new FormData(form));

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    const blocked = result.action === "BLOCK";
    resultChip.className = `result-chip ${blocked ? "block" : "approve"}`;
    resultChip.textContent = blocked ? "HIGH RISK" : "CLEAR";
    actionText.textContent = `${result.action} the transaction`;
    probabilityText.textContent = `${(result.fraud_probability * 100).toFixed(2)}%`;
    thresholdText.textContent = `${(result.threshold_used * 100).toFixed(2)}%`;
    modelText.textContent = result.model_name;
  } catch (error) {
    resultChip.className = "result-chip block";
    resultChip.textContent = "ERROR";
    actionText.textContent = "Prediction failed";
    probabilityText.textContent = "Check API logs";
    thresholdText.textContent = "--";
    modelText.textContent = error.message;
  }
});
