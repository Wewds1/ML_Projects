const form = document.getElementById("predict-form");
const sampleButton = document.getElementById("load-sample");
const statusText = document.getElementById("status-text");

function setValue(id, value) {
  document.getElementById(id).textContent = value;
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  const data = await response.json();
  document.getElementById("model-metadata").innerHTML = `
    <span class="label">Best experiment</span>
    <strong>${data.best_experiment.replaceAll("_", " ")}</strong>
    <p>RMSE (log): ${data.experiments[data.best_experiment].rmse_log.toFixed(3)}<br>
    R²: ${data.experiments[data.best_experiment].r2.toFixed(3)}</p>
  `;
}

async function loadSample() {
  const response = await fetch("/api/sample");
  const sample = await response.json();
  Object.entries(sample).forEach(([key, value]) => {
    const field = form.elements.namedItem(key);
    if (field) {
      field.value = value ?? "";
    }
  });
}

function toPayload(formData) {
  const numericFields = [
    "customer_age",
    "tenure_months",
    "monthly_spend",
    "login_freq_monthly",
    "feature_adoption",
    "support_tickets_6m",
    "nps_score",
    "payment_failures_6m",
    "referrals_made",
    "days_since_login",
  ];
  const payload = Object.fromEntries(formData.entries());
  numericFields.forEach((field) => {
    payload[field] = payload[field] === "" ? null : Number(payload[field]);
  });
  if (payload.customer_feedback === "") {
    payload.customer_feedback = null;
  }
  return payload;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusText.textContent = "Scoring";
  const payload = toPayload(new FormData(form));

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();

  setValue("clv-value", `$${result.clv_12m_estimate_usd.toLocaleString()}`);
  setValue(
    "prediction-detail",
    `Log prediction ${result.log_clv_prediction}, with ${result.churn_risk_flag ? "churn" : "no churn"} language detected.`
  );
  setValue("sentiment-value", result.sentiment_polarity.toFixed(3));
  setValue("subjectivity-value", result.sentiment_subjectivity.toFixed(3));
  setValue("churn-value", String(result.churn_risk_flag));
  setValue("praise-value", String(result.praise_language_flag));
  statusText.textContent = "Complete";
});

sampleButton.addEventListener("click", loadSample);
loadMetadata();

