/* ---------------------------------------------------------------------------
   app.js
   ---------------------------------------------------------------------------
   Presentation + API calls ONLY.

   This file reads form values, POSTs them as JSON, and displays whatever the
   API sends back. It has NO knowledge of how the prediction is produced: no
   pandas, no scikit-learn, no CSV column names, no model files, no accuracy
   maths. Swap the model on the backend and this file is unaffected.

   The three derived numbers it shows are pure unit/format conversions of
   values the API already returned:
     * monthly          = annual / 12
     * "avg. error" etc = the metrics from GET /model-info, formatted
   Nothing is predicted, scored or inferred here.
   --------------------------------------------------------------------------- */

/* Where the backend lives. */
const API_URL = "https://medical-cost-prediction-1-xs28.onrender.com/api/v1/predict";
const META_URL = "https://medical-cost-prediction-1-xs28.onrender.com/api/v1/model-info";

/* --- Elements --- */
const form = document.getElementById("quote-form");
const submit = document.getElementById("submit");
const errorBox = document.getElementById("form-error");
const result = document.getElementById("result");
const resultEmpty = document.getElementById("result-empty");
const resultBody = document.getElementById("result-body");

const age = document.getElementById("age");
const bmi = document.getElementById("bmi");
const bmiRange = document.getElementById("bmi-range");
const bmiBadge = document.getElementById("bmi-badge");
const children = document.getElementById("children");
const region = document.getElementById("region");

/* --- Formatting --- */
const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

/* Readable labels for the echoed input fields. */
const LABELS = {
  age: "Age",
  sex: "Sex",
  bmi: "BMI",
  children: "Children",
  smoker: "Smoker",
  region: "Region",
};

/* ------------------------------------------------------------------ */
/* BMI: number box + slider stay in sync, plus a plain-language band.   */
/* ------------------------------------------------------------------ */
function bmiBand(value) {
  if (value < 18.5) return ["Underweight", ""];
  if (value < 25) return ["Healthy weight", ""];
  if (value < 30) return ["Overweight", "overweight"];
  return ["Obese", "overweight"];
}

function syncBmi(source) {
  /* Mirror the edited control into the other one, but never write back to
     the control the user is currently dragging or typing in. */
  if (source === bmiRange) bmi.value = bmiRange.value;
  else bmiRange.value = bmi.value;

  const value = Number(bmi.value);
  if (Number.isFinite(value)) {
    bmiBadge.textContent = bmiBand(value)[0];
    bmiBadge.dataset.level = bmiBand(value)[1];
  }
}

bmi.addEventListener("input", () => syncBmi(bmi));
bmiRange.addEventListener("input", () => syncBmi(bmiRange));

/* ------------------------------------------------------------------ */
/* Children: -/+ stepper                                               */
/* ------------------------------------------------------------------ */
document.querySelectorAll(".stepper__btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const step = Number(btn.dataset.step);
    const next = Math.min(10, Math.max(0, (Number(children.value) || 0) + step));
    children.value = next;
    btn.disabled = (step < 0 && next === 0) || (step > 0 && next === 10);
  });
});

/* ------------------------------------------------------------------ */
/* Prefill the form from the URL, e.g.                                 */
/*   /?age=45&sex=male&bmi=27.5&children=1&smoker=no&region=southwest   */
/* ------------------------------------------------------------------ */
function prefillFromUrl() {
  const params = new URLSearchParams(window.location.search);

  const setNumber = (el, raw, min, max) => {
    if (raw === null) return;
    const n = Number(raw);
    if (!Number.isFinite(n)) return;
    el.value = Math.min(max, Math.max(min, n));   // clamp to the API's own limits
  };

  setNumber(age, params.get("age"), 18, 100);
  setNumber(bmi, params.get("bmi"), 10, 70);
  setNumber(children, params.get("children"), 0, 10);

  const sex = params.get("sex");
  if (sex) {
    const radio = form.querySelector(`input[name="sex"][value="${sex.trim().toLowerCase()}"]`);
    if (radio) radio.checked = true;
  }

  const smoker = params.get("smoker");
  if (smoker) {
    const radio = form.querySelector(`input[name="smoker"][value="${smoker.trim().toLowerCase()}"]`);
    if (radio) radio.checked = true;
  }

  const reg = params.get("region");
  if (reg) {
    const option = [...region.options].find(
      (o) => o.value === reg.trim().toLowerCase()
    );
    if (option) region.value = option.value;
  }

  /* Keep the +/- buttons in the right disabled state. */
  document.querySelectorAll(".stepper__btn").forEach((btn) => {
    const step = Number(btn.dataset.step);
    const n = Number(children.value) || 0;
    btn.disabled = (step < 0 && n <= 0) || (step > 0 && n >= 10);
  });

  syncBmi(bmi);
}

/* Mirror the current values back into the address bar after a prediction. */
function syncUrl() {
  const data = new FormData(form);
  const params = new URLSearchParams();
  for (const key of ["age", "sex", "bmi", "children", "smoker", "region"]) {
    if (data.get(key) !== null) params.set(key, data.get(key));
  }
  history.replaceState(null, "", `?${params.toString()}`);
}

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */
function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

/* ------------------------------------------------------------------ */
/* Collect + render                                                    */
/* ------------------------------------------------------------------ */
function collect() {
  const data = new FormData(form);
  return {
    age: Number(data.get("age")),
    sex: data.get("sex"),
    bmi: Number(data.get("bmi")),
    children: Number(data.get("children")),
    smoker: data.get("smoker"),
    region: data.get("region"),
  };
}

function renderInputs(inputs) {
  /* Build the list from the API's own echo of what it priced. */
  const dl = document.getElementById("out-inputs");
  dl.replaceChildren();
  for (const key of Object.keys(LABELS)) {
    const dt = document.createElement("dt");
    dt.textContent = LABELS[key];
    const dd = document.createElement("dd");
    dd.textContent = String(inputs[key]);
    dl.append(dt, dd);
  }
}

function render(data) {
  /* The amount, exactly as the API returned it. */
  document.getElementById("out-amount").textContent =
    money.format(data.predicted_medical_charges_usd);

  /* Same number restated per month - a unit conversion, nothing more. */
  document.getElementById("out-monthly").textContent =
    `${money.format(data.predicted_medical_charges_usd / 12)} per month`;

  /* Which model answered, straight from the response. */
  document.getElementById("out-tier-name").textContent = data.model;
  document.getElementById("out-tier-note").textContent =
    `version ${data.model_version} \u00b7 ${data.currency}`;

  renderInputs(data.input);

  /* Reveal the result with a short animation. */
  resultEmpty.hidden = true;
  resultBody.hidden = false;
  result.classList.remove("is-entering");
  void result.offsetWidth;          // force a reflow so the animation restarts
  result.classList.add("is-entering");
}

/* Read a response body as JSON without throwing. */
async function readJson(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Submit                                                              */
/* ------------------------------------------------------------------ */
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  submit.classList.add("is-loading");
  submit.disabled = true;

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collect()),
    });

    const data = await readJson(res);

    if (!res.ok) {
      /* The API always answers errors as { success:false, error:{code,message} }. */
      const detail = data && data.error ? data.error : {};
      throw new Error(detail.message || `The server returned HTTP ${res.status}.`);
    }
    if (!data || data.success !== true) {
      throw new Error("The API returned an unexpected response.");
    }

    render(data);
    syncUrl();
  } catch (err) {
    showError(err.message || "Could not reach the service. Is the backend running on port 8001?");
  } finally {
    submit.classList.remove("is-loading");
    submit.disabled = false;
  }
});

/* ------------------------------------------------------------------ */
/* Copy summary to the clipboard                                       */
/* ------------------------------------------------------------------ */
document.getElementById("copy-btn").addEventListener("click", async (event) => {
  const text = [
    "MediQuote - medical cost estimate",
    `Estimated annual charges: ${document.getElementById("out-amount").textContent}`,
    document.getElementById("out-monthly").textContent,
    `Model: ${document.getElementById("out-tier-name").textContent}`,
  ].join("\n");

  try {
    await navigator.clipboard.writeText(text);
    const btn = event.currentTarget;
    const original = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => (btn.textContent = original), 1600);
  } catch {
    /* clipboard unavailable - ignore */
  }
});

/* ------------------------------------------------------------------ */
/* Model metrics: GET /api/v1/model-info                                */
/* Display only - the numbers are formatted, never recomputed.          */
/* ------------------------------------------------------------------ */
async function loadModelInfo() {
  try {
    const res = await fetch(META_URL);
    const data = await readJson(res);
    if (!res.ok || !data || data.success !== true) return;

    const m = data.metrics;

    /* Hero strip */
    document.getElementById("stat-r2").textContent = Number(m.R2).toFixed(3);
    document.getElementById("stat-mae").textContent = money.format(m.MAE);
    document.getElementById("stat-rmse").textContent = money.format(m.RMSE);

    /* Diagnostics tiles */
    document.getElementById("m-r2").textContent = Number(m.R2).toFixed(4);
    document.getElementById("m-mae").textContent = money.format(m.MAE);
    document.getElementById("m-rmse").textContent = money.format(m.RMSE);
    document.getElementById("m-model").textContent = data.model;
    document.getElementById("m-version").textContent = `version ${data.model_version}`;

    /* Bar simply visualises the R-squared value the API returned. */
    document.getElementById("m-r2-bar").style.width =
      `${Math.max(0, Math.min(1, Number(m.R2))) * 100}%`;

    /* Footer */
    document.getElementById("foot-model").textContent =
      `Model: ${data.model} \u00b7 R\u00b2 ${Number(m.R2).toFixed(4)} ` +
      `\u00b7 mean absolute error ${money.format(m.MAE)}`;
  } catch {
    /* Backend not running - the page still works, the tiles stay blank. */
  }
}

/* --- Start up --- */
prefillFromUrl();
loadModelInfo();