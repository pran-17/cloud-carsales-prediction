import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for server
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, redirect, url_for, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# HTML TEMPLATES (inline for simplicity)
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Sales Prediction</title>
</head>
<body>
    <h1>Car Sales Prediction</h1>
    <form method="post">
        <label for="brand">Select Brand:</label>
        <select name="brand" id="brand" required>
            <option value="" disabled selected>Select a brand</option>
            {% for b in brands %}
                <option value="{{ b }}">{{ b }}</option>
            {% endfor %}
        </select>
        <button type="submit">Next</button>
    </form>
</body>
</html>
"""

MODEL_FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Sales Prediction</title>
</head>
<body>
    <h1>Car Sales Prediction</h1>
    <h2>Brand: {{ brand }}</h2>
    {% if error %}
        <p style="color:red;">{{ error }}</p>
    {% endif %}
    <form method="post">
        <input type="hidden" name="brand" value="{{ brand }}">
        <label for="model">Select Model:</label>
        <select name="model" id="model" required>
            <option value="" disabled selected>Select a model</option>
            {% for m in models %}
                <option value="{{ m }}">{{ m }}</option>
            {% endfor %}
        </select>
        <br><br>
        <label for="year">Year for prediction (e.g., 2025):</label>
        <input type="number" name="year" id="year" required min="2000" max="2100">
        <br><br>
        <label for="month">Month (1-12):</label>
        <input type="number" name="month" id="month" required min="1" max="12">
        <br><br>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Sales Prediction Result</title>
</head>
<body>
    <h1>Car Sales Prediction Result</h1>

    {% if error %}
        <p style="color:red;">{{ error }}</p>
    {% else %}
        <h2>{{ car_brand }} {{ car_model }} - {{ year }} / {{ month }}</h2>

        <h3>Predictions</h3>
        <p><b>Predicted Price:</b> ₹{{ "{:,.2f}".format(predicted_price) }}</p>
        {% if has_units %}
            <p><b>Predicted Units Sold:</b> {{ predicted_units }} units</p>
        {% else %}
            <p><b>Units Sold Prediction:</b> Not available (no 'units_sold' column or insufficient data).</p>
        {% endif %}
        <p><b>LDA Classification Accuracy (High/Low price):</b> {{ lda_accuracy }}%</p>

        <h3>Plots</h3>

        <h4>1. Actual vs Predicted Prices</h4>
        <img src="data:image/png;base64,{{ scatter_img }}" alt="Actual vs Predicted">

        <h4>2. Price Trend Over Time</h4>
        <img src="data:image/png;base64,{{ trend_img }}" alt="Price Trend">

        <h4>3. Price Distribution</h4>
        <img src="data:image/png;base64,{{ hist_img }}" alt="Price Histogram">
    {% endif %}

    <br><br>
    <a href="{{ url_for('index') }}">Start Again</a>
</body>
</html>
"""

# -------------------- HELPER: PLOT TO BASE64 --------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

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

    # Scatter plot: Actual vs Predicted
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    ax1.set_xlabel("Actual Price")
    ax1.set_ylabel("Predicted Price")
    ax1.set_title("Actual vs Predicted Prices")
    ax1.grid(True)
    scatter_img = fig_to_base64(fig1)

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

    # -------------------- VISUALIZATIONS --------------------
    # Price trend over time
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=model_data['date'], y=model_data['price'], ax=ax2)
    pred_date = pd.Timestamp(year_input, month_input, 15)
    ax2.axvline(pred_date, color='red', linestyle='--', label='Prediction Date')
    ax2.set_title(f"Price Trend - {car_brand} {car_model}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True)
    trend_img = fig_to_base64(fig2)

    # Price distribution
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(model_data['price'], kde=True, bins=20, ax=ax3)
    ax3.set_title("Price Distribution")
    ax3.set_xlabel("Price")
    ax3.set_ylabel("Frequency")
    ax3.grid(True)
    hist_img = fig_to_base64(fig3)

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
        "scatter_img": scatter_img,
        "trend_img": trend_img,
        "hist_img": hist_img,
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
    app.run(debug=True)
