import pandas as pd
import numpy as np

from flask import Flask, request, redirect, url_for, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# -------------------- LOAD DATA ONCE --------------------
file_path = "carsales.csv"
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.lower().str.strip()

required_columns = {'brand', 'model', 'month', 'price'}
if not required_columns.issubset(data.columns):
    missing = required_columns - set(data.columns)
    raise KeyError(f"Missing required columns: {missing}")

available_brands = sorted(data['brand'].dropna().unique())

# -------------------- FLASK APP --------------------
app = Flask(__name__)

SHARED_META = """
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
"""

SHARED_STYLES = """
<style>
:root {
  --bg: #f0f4f8;
  --bg-deep: #e2e8f0;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --accent-soft: #dbeafe;
  --success: #059669;
  --success-soft: #d1fae5;
  --warning: #d97706;
  --danger: #dc2626;
  --radius: 16px;
  --shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
  --shadow-lg: 0 20px 50px rgba(15, 23, 42, 0.12);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: "Plus Jakarta Sans", system-ui, sans-serif;
  background: var(--bg);
  background-image:
    radial-gradient(ellipse 900px 500px at 15% -10%, rgba(37, 99, 235, 0.12), transparent),
    radial-gradient(ellipse 700px 400px at 100% 20%, rgba(5, 150, 105, 0.08), transparent);
  color: var(--text);
  line-height: 1.55;
}
.shell {
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 1.25rem 3rem;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1.75rem;
}
.logo {
  font-weight: 700;
  font-size: 1.35rem;
  letter-spacing: -0.02em;
  color: var(--text);
}
.logo span { color: var(--accent); }
.card {
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow-lg);
  padding: 1.75rem 1.75rem 2rem;
  border: 1px solid rgba(148, 163, 184, 0.15);
}
.card h1 {
  margin: 0 0 0.35rem;
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.lead {
  margin: 0 0 1.5rem;
  color: var(--muted);
  font-size: 0.95rem;
}
.form-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem 1.1rem;
  align-items: end;
}
@media (max-width: 900px) {
  .form-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 520px) {
  .form-grid { grid-template-columns: 1fr; }
}
.field { display: flex; flex-direction: column; gap: 0.4rem; }
.field label {
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.field-brand .brand-chip {
  display: flex;
  align-items: center;
  min-height: 48px;
  padding: 0 1rem;
  background: var(--accent-soft);
  color: var(--accent);
  border-radius: 12px;
  font-weight: 600;
  font-size: 1rem;
  border: 1px solid rgba(37, 99, 235, 0.2);
}
select, input[type="number"] {
  width: 100%;
  padding: 0.65rem 0.85rem;
  font-size: 1rem;
  font-family: inherit;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  background: #fff;
  color: var(--text);
  transition: border-color 0.15s, box-shadow 0.15s;
}
select:focus, input[type="number"]:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 1.5rem;
  align-items: center;
}
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1.35rem;
  font-size: 0.95rem;
  font-weight: 600;
  font-family: inherit;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: transform 0.12s, box-shadow 0.12s, background 0.12s;
}
.btn-primary {
  background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%);
  color: #fff;
  box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
}
.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
}
.btn-ghost {
  background: transparent;
  color: var(--muted);
  border: 1px solid #cbd5e1;
}
.btn-ghost:hover {
  background: var(--bg-deep);
  color: var(--text);
}
.alert {
  padding: 0.85rem 1rem;
  border-radius: 12px;
  margin-bottom: 1rem;
  font-size: 0.9rem;
}
.alert-error {
  background: #fef2f2;
  color: #b91c1c;
  border: 1px solid #fecaca;
}
.pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.75rem 0 1.5rem;
}
.pill {
  display: inline-flex;
  align-items: center;
  padding: 0.35rem 0.85rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  background: var(--bg-deep);
  color: var(--text);
  border: 1px solid #cbd5e1;
}
.pill-accent { background: var(--accent-soft); color: var(--accent); border-color: rgba(37,99,235,0.25); }
.metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
@media (max-width: 720px) {
  .metrics { grid-template-columns: 1fr; }
}
.metric {
  background: linear-gradient(145deg, #fff 0%, #f8fafc 100%);
  border-radius: 14px;
  padding: 1.15rem 1.25rem;
  border: 1px solid #e2e8f0;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.metric:hover {
  border-color: rgba(37, 99, 235, 0.25);
  box-shadow: var(--shadow);
}
.metric-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  margin-bottom: 0.35rem;
}
.metric-value {
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
}
.metric-value.success { color: var(--success); }
.chart-section-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0 0 1rem;
  letter-spacing: -0.02em;
}
.chart-card {
  background: #fff;
  border-radius: 14px;
  padding: 1.25rem;
  margin-bottom: 1.25rem;
  border: 1px solid #e2e8f0;
  box-shadow: var(--shadow);
}
.chart-card h3 {
  margin: 0 0 0.75rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--muted);
}
.chart-wrap {
  position: relative;
  height: 320px;
  max-height: 55vh;
}
.chart-wrap.tall { height: 360px; }
@media (max-width: 600px) {
  .chart-wrap { height: 260px; }
}
.step-badge {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
  background: var(--accent-soft);
  padding: 0.25rem 0.6rem;
  border-radius: 6px;
  margin-bottom: 0.5rem;
}
</style>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
""" + SHARED_META + """
    <title>Car Sales Prediction</title>
""" + SHARED_STYLES + """
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="logo">Sales<span>Insight</span></div>
    </div>
    <div class="card">
      <span class="step-badge">Step 1 of 2</span>
      <h1>Car Sales Prediction</h1>
      <p class="lead">Choose a car brand to continue. You will pick model, year, and month on the next screen.</p>
      <form method="post">
        <div class="form-grid" style="grid-template-columns: 1fr auto; align-items: end;">
          <div class="field">
            <label for="brand">Car brand</label>
            <select name="brand" id="brand" required>
              <option value="" disabled selected>Select a brand</option>
              {% for b in brands %}
              <option value="{{ b }}">{{ b }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="field" style="min-width: 140px;">
            <label style="opacity:0;">.</label>
            <button type="submit" class="btn btn-primary" style="width:100%;">Continue</button>
          </div>
        </div>
      </form>
    </div>
  </div>
</body>
</html>
"""

MODEL_FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
""" + SHARED_META + """
    <title>Car Sales Prediction</title>
""" + SHARED_STYLES + """
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="logo">Sales<span>Insight</span></div>
      <a class="btn btn-ghost" href="{{ url_for('index') }}">← Change brand</a>
    </div>
    <div class="card">
      <span class="step-badge">Step 2 of 2</span>
      <h1>Refine your prediction</h1>
      <p class="lead">Fields follow the order: brand → model → Predictionyear → month. Then run the forecast.</p>
      {% if error %}
      <div class="alert alert-error">{{ error }}</div>
      {% endif %}
      <form method="post">
        <input type="hidden" name="brand" value="{{ brand }}">
        <div class="form-grid">
          <div class="field field-brand">
            <label>Car brand</label>
            <div class="brand-chip" title="Selected brand">{{ brand }}</div>
          </div>
          <div class="field">
            <label for="model">Model</label>
            <select name="model" id="model" required>
              <option value="" disabled selected>Select model</option>
              {% for m in models %}
              <option value="{{ m }}">{{ m }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="field">
            <label for="year">Prediction Year</label>
            <input type="number" name="year" id="year" required min="2000" max="2100" placeholder="e.g. 2026" value="2026">
          </div>
          <div class="field">
            <label for="month">Month</label>
            <select name="month" id="month" required>
              <option value="" disabled selected>Month</option>
              {% set month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] %}
              {% for m in range(1, 13) %}
              <option value="{{ m }}">{{ m }} — {{ month_names[m - 1] }}</option>
              {% endfor %}
            </select>
          </div>
        </div>
        <div class="actions">
          <button type="submit" class="btn btn-primary">Predict sales</button>
        </div>
      </form>
    </div>
  </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
""" + SHARED_META + """
    <title>Prediction Result</title>
""" + SHARED_STYLES + """
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="logo">Drive<span>Insight</span></div>
      <a class="btn btn-ghost" href="{{ url_for('index') }}">New prediction</a>
    </div>
    <div class="card">
      {% if error %}
      <h1>Something went wrong</h1>
      <p class="lead">We could not complete the forecast.</p>
      <div class="alert alert-error">{{ error }}</div>
      <div class="actions">
        <a class="btn btn-primary" href="{{ url_for('index') }}">Start again</a>
      </div>
      {% else %}
      <h1>Your forecast</h1>
      <p class="lead">Hover the charts to explore values. Red dashed line marks your selected prediction date on the trend.</p>
      <div class="pills">
        <span class="pill pill-accent">{{ car_brand }}</span>
        <span class="pill">{{ car_model }}</span>
        <span class="pill">Year {{ year }}</span>
        <span class="pill">Month {{ month }}</span>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Predicted price</div>
          <div class="metric-value">₹{{ "{:,.2f}".format(predicted_price) }}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Predicted units sold</div>
          <div class="metric-value {% if has_units %}success{% endif %}">
            {% if has_units %}{{ predicted_units }} units{% else %}—{% endif %}
          </div>
          {% if not has_units %}
          <div style="font-size:0.8rem;color:var(--muted);margin-top:0.35rem;">Not available for this dataset</div>
          {% endif %}
        </div>
        <div class="metric">
          <div class="metric-label">LDA accuracy (high / low price)</div>
          <div class="metric-value success">{{ lda_accuracy }}%</div>
        </div>
      </div>
      <h2 class="chart-section-title">Interactive charts</h2>
      <div class="chart-card">
        <h3>Actual vs predicted prices</h3>
        <div class="chart-wrap"><canvas id="chartScatter"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Price trend over time</h3>
        <div class="chart-wrap tall"><canvas id="chartTrend"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Price distribution</h3>
        <div class="chart-wrap"><canvas id="chartHist"></canvas></div>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
      <script>
(function() {
  const rupee = (n) => "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 });
  const scatterRaw = {{ scatter_chart|tojson }};
  const trendRaw = {{ trend_chart|tojson }};
  const histRaw = {{ hist_chart|tojson }};
  const predTs = {{ pred_ts_ms|tojson }};
  const predictedPrice = {{ predicted_price|tojson }};

  const gridColor = "rgba(148, 163, 184, 0.35)";
  const fontFamily = "'Plus Jakarta Sans', system-ui, sans-serif";

  new Chart(document.getElementById("chartScatter"), {
    type: "scatter",
    data: {
      datasets: [{
        label: "Test points",
        data: scatterRaw,
        backgroundColor: "rgba(37, 99, 235, 0.55)",
        borderColor: "rgba(37, 99, 235, 1)",
        borderWidth: 1,
        pointRadius: 6,
        pointHoverRadius: 9
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw;
              return "Actual: " + rupee(p.x) + " · Predicted: " + rupee(p.y);
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Actual price", font: { family: fontFamily, weight: "600" } },
          grid: { color: gridColor },
          ticks: { callback: (v) => rupee(v) }
        },
        y: {
          title: { display: true, text: "Predicted price", font: { family: fontFamily, weight: "600" } },
          grid: { color: gridColor },
          ticks: { callback: (v) => rupee(v) }
        }
      }
    }
  });

  const trendYs = trendRaw.map((d) => d.y);
  const yMin = Math.min(...trendYs, predictedPrice) * 0.97;
  const yMax = Math.max(...trendYs, predictedPrice) * 1.03;

  new Chart(document.getElementById("chartTrend"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "Historical price",
          data: trendRaw,
          parsing: { xAxisKey: "x", yAxisKey: "y" },
          borderColor: "rgba(37, 99, 235, 1)",
          backgroundColor: "rgba(37, 99, 235, 0.08)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          pointHoverRadius: 5,
          borderWidth: 2
        },
        {
          label: "Prediction date",
          data: [{ x: predTs, y: yMin }, { x: predTs, y: yMax }],
          borderColor: "rgba(239, 68, 68, 0.9)",
          borderWidth: 2,
          borderDash: [7, 5],
          pointRadius: 0,
          tension: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.datasetIndex === 1) return "Your selected date";
              const p = ctx.raw;
              const d = new Date(p.x);
              return d.toLocaleDateString("en-IN", { year: "numeric", month: "short", day: "numeric" })
                + " · " + rupee(p.y);
            }
          }
        }
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Date", font: { family: fontFamily, weight: "600" } },
          grid: { color: gridColor },
          ticks: {
            maxTicksLimit: 8,
            callback: (v) => new Date(v).toLocaleDateString("en-IN", { year: "2-digit", month: "short" })
          }
        },
        y: {
          title: { display: true, text: "Price", font: { family: fontFamily, weight: "600" } },
          grid: { color: gridColor },
          min: yMin,
          max: yMax,
          ticks: { callback: (v) => rupee(v) }
        }
      }
    }
  });

  const histLabels = histRaw.map((h) => rupee(h.x));
  const histCounts = histRaw.map((h) => h.y);
  new Chart(document.getElementById("chartHist"), {
    type: "bar",
    data: {
      labels: histLabels,
      datasets: [{
        label: "Frequency",
        data: histCounts,
        backgroundColor: "rgba(5, 150, 105, 0.45)",
        borderColor: "rgba(5, 150, 105, 1)",
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => "Bin centre ~ " + ctx.label + " · Count: " + ctx.raw
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Price (bin centre)", font: { family: fontFamily, weight: "600" } },
          grid: { display: false },
          ticks: { maxRotation: 45, minRotation: 0, autoSkip: true, maxTicksLimit: 12 }
        },
        y: {
          title: { display: true, text: "Count", font: { family: fontFamily, weight: "600" } },
          grid: { color: gridColor },
          beginAtZero: true,
          ticks: { precision: 0 }
        }
      }
    }
  });
})();
      </script>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

# -------------------- CORE ANALYSIS FUNCTION --------------------
def run_analysis(data, car_brand, car_model, year_input, month_input):
    # Filter by brand and model
    brand_data = data[data['brand'].str.lower() == car_brand.lower()]
    if brand_data.empty:
        return {"error": "No data found for the selected brand."}

    model_data = brand_data[brand_data['model'].str.lower() == car_model.lower()]
    if model_data.empty:
        return {"error": "No data found for the selected model."}

    # Parse month column to date
    model_data = model_data.copy()  # avoid SettingWithCopyWarnings
    model_data['date'] = pd.to_datetime(model_data['month'] + ' 15', errors='coerce')
    model_data = model_data.dropna(subset=['date'])
    model_data['year'] = model_data['date'].dt.year
    model_data['month_num'] = model_data['date'].dt.month  # keep numeric month separate

    # Ensure data integrity
    model_data = model_data.dropna(subset=['year', 'month_num', 'price'])
    model_data = shuffle(model_data, random_state=42)

    if len(model_data) < 5:
        return {"error": "Not enough data for this brand/model. Try another combination."}

    # -------------------- PRICE PREDICTION --------------------
    X = model_data[['year', 'month_num']]
    y = model_data['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scatter_chart = [
        {"x": float(a), "y": float(p)}
        for a, p in zip(np.asarray(y_test).ravel(), np.asarray(y_pred).ravel())
    ]

    # User prediction
    input_data = pd.DataFrame({'year': [year_input], 'month_num': [month_input]})
    predicted_price = model.predict(input_data)[0]

    # -------------------- CLASSIFICATION (High/Low Price) --------------------
    price_threshold = model_data['price'].median()
    model_data['price_category'] = np.where(
        model_data['price'] > price_threshold, 'High', 'Low'
    )
    model_data['price_category_encoded'] = model_data['price_category'].map(
        {'Low': 0, 'High': 1}
    )

    X_lda = model_data[['year', 'month_num', 'price']]
    y_lda = model_data['price_category_encoded']
    X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(
        X_lda, y_lda, test_size=0.2, random_state=42
    )

    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train_lda, y_train_lda)
    y_pred_lda = lda_model.predict(X_test_lda)
    lda_acc = round(accuracy_score(y_test_lda, y_pred_lda) * 100, 2)

    # -------------------- UNITS SOLD (optional) --------------------
    has_units = False
    predicted_units = None

    if 'units_sold' in model_data.columns:
        units_data = model_data.dropna(subset=['units_sold'])
        if len(units_data) >= 5:
            X_sales = units_data[['year', 'month_num', 'price']]
            y_sales = units_data['units_sold']
            Xs_train, Xs_test, ys_train, ys_test = train_test_split(
                X_sales, y_sales, test_size=0.2, random_state=42
            )

            sales_model = LinearRegression()
            sales_model.fit(Xs_train, ys_train)

            predicted_units_val = sales_model.predict(pd.DataFrame({
                'year': [year_input],
                'month_num': [month_input],
                'price': [predicted_price]
            }))[0]

            has_units = True
            predicted_units = int(round(predicted_units_val))

    # -------------------- CHART DATA (for interactive front-end) --------------------
    trend_df = model_data.sort_values("date")
    trend_chart = [
        {"x": int(pd.Timestamp(d).timestamp() * 1000), "y": float(p)}
        for d, p in zip(trend_df["date"], trend_df["price"])
    ]
    pred_ts_ms = int(pd.Timestamp(year_input, month_input, 15).timestamp() * 1000)

    prices = model_data["price"].to_numpy()
    counts, bin_edges = np.histogram(prices, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_chart = [
        {"x": float(c), "y": int(n)} for c, n in zip(bin_centers, counts)
    ]

    return {
        "error": None,
        "car_brand": car_brand,
        "car_model": car_model,
        "year": year_input,
        "month": month_input,
        "predicted_price": predicted_price,
        "lda_accuracy": lda_acc,
        "has_units": has_units,
        "predicted_units": predicted_units,
        "scatter_chart": scatter_chart,
        "trend_chart": trend_chart,
        "hist_chart": hist_chart,
        "pred_ts_ms": pred_ts_ms,
    }

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        brand = request.form.get("brand")
        return redirect(url_for("model_form", brand=brand))
    return render_template_string(INDEX_HTML, brands=available_brands)

@app.route("/model", methods=["GET", "POST"])
def model_form():
    if request.method == "GET":
        brand = request.args.get("brand")
    else:
        brand = request.form.get("brand")

    brand_data = data[data['brand'].str.lower() == str(brand).lower()]
    models = sorted(brand_data['model'].dropna().unique())

    if request.method == "POST":
        model_name = request.form.get("model")
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))
        result = run_analysis(data, brand, model_name, year, month)
        return render_template_string(RESULT_HTML, **result)

    if len(models) == 0:
        error = "No models found for this brand. Try another brand."
    else:
        error = None

    return render_template_string(
        MODEL_FORM_HTML,
        brand=brand,
        models=models,
        error=error
    )

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
