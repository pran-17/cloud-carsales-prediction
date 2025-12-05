import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysisgit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
file_path = "carsales.csv"
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.lower().str.strip()

# Check required columns
required_columns = {'brand', 'model', 'month', 'price'}
if not required_columns.issubset(data.columns):
    raise KeyError(f"Missing required columns: {required_columns - set(data.columns)}")

# Input: brand
available_brands = data['brand'].dropna().unique()
print(f"Available car brands: {', '.join(available_brands)}")
car_brand = input("Enter the car brand from the above list: ")

brand_data = data[data['brand'].str.lower() == car_brand.lower()]
if brand_data.empty:
    print("No data found for the selected brand.")
    exit()

# Input: model
available_models = brand_data['model'].dropna().unique()
print(f"Available models for {car_brand}: {', '.join(available_models)}")
car_model = input("Enter the car model from the above list: ")

model_data = brand_data[brand_data['model'].str.lower() == car_model.lower()]
if model_data.empty:
    print("No data found for the selected model.")
    exit()

# Parse month
model_data['date'] = pd.to_datetime(model_data['month'] + ' 15', errors='coerce')
model_data = model_data.dropna(subset=['date'])
model_data['year'] = model_data['date'].dt.year
model_data['month'] = model_data['date'].dt.month

# Ensure data integrity
model_data = model_data.dropna(subset=['year', 'month', 'price'])
model_data = shuffle(model_data, random_state=42)

print(f"\nFiltered data size: {model_data.shape[0]}")
print(f"Duplicate rows: {model_data.duplicated().sum()}")
print(model_data[['price']].describe())

if len(model_data) < 5:
    print("Not enough data. Try another brand/model.")
    exit()

# -------------------- PRICE PREDICTION --------------------
X = model_data[['year', 'month']]
y = model_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# User prediction input
year_input = int(input("Enter the year for prediction (e.g., 2025): "))
month_input = int(input("Enter the month for prediction (1-12): "))
input_data = pd.DataFrame({'year': [year_input], 'month': [month_input]})
predicted_price = model.predict(input_data)
print(f"\nPredicted price for {car_brand} {car_model} in {year_input}-{month_input}: ₹{predicted_price[0]:,.2f}\n")

# -------------------- CLASSIFICATION --------------------
price_threshold = model_data['price'].median()
model_data['price_category'] = np.where(model_data['price'] > price_threshold, 'High', 'Low')
model_data['price_category_encoded'] = model_data['price_category'].map({'Low': 0, 'High': 1})

X_lda = model_data[['year', 'month', 'price']]
y_lda = model_data['price_category_encoded']
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y_lda, test_size=0.2, random_state=42)

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_lda, y_train_lda)
y_pred_lda = lda_model.predict(X_test_lda)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_lda, y_pred_lda))

print("\nClassification Report:")
print(classification_report(y_test_lda, y_pred_lda, target_names=['Low', 'High'], zero_division=0))
print(f"\nLDA Accuracy: {accuracy_score(y_test_lda, y_pred_lda) * 100:.2f}%\n")

# Scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=model_data, x='price', y='month', hue='price_category', palette='Set1', s=60)
plt.title("Price Category Distribution")
plt.xlabel("Price")
plt.ylabel("Month")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- UNITS SOLD --------------------
if 'units_sold' in model_data.columns:
    model_data = model_data.dropna(subset=['units_sold'])
    if len(model_data) < 5:
        print("Not enough data for units_sold prediction.")
    else:
        X_sales = model_data[['year', 'month', 'price']]
        y_sales = model_data['units_sold']
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)

        sales_model = LinearRegression()
        sales_model.fit(Xs_train, ys_train)

        predicted_units = sales_model.predict(pd.DataFrame({
            'year': [year_input],
            'month': [month_input],
            'price': [predicted_price[0]]
        }))

        print(f"\nPredicted units sold: {int(predicted_units[0])} units\n")
else:
    print("\n'units_sold' column not found. Skipping prediction.\n")

# -------------------- VISUALIZATIONS --------------------
# Price trend
plt.figure(figsize=(10, 5))
sns.lineplot(x=model_data['date'], y=model_data['price'])
plt.axvline(pd.Timestamp(year_input, month_input, 15), color='red', linestyle='--', label='Prediction')
plt.title(f"Price Trend - {car_brand} {car_model}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Monthly avg trend
plt.figure(figsize=(10, 5))
monthly_avg = model_data.groupby(['year', 'month'])['price'].mean().reset_index()
monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=15))
sns.lineplot(data=monthly_avg, x='date', y='price')
plt.title(f"Monthly Avg Price - {car_brand} {car_model}")
plt.xlabel("Date")
plt.ylabel("Avg Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(8, 4))
sns.histplot(model_data['price'], kde=True, bins=20, color='skyblue')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Price vs Units Sold
if 'units_sold' in model_data.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=model_data, x='price', y='units_sold', hue='year', palette='viridis', s=60)
    plt.title(f"Price vs Units Sold - {car_brand} {car_model}")
    plt.xlabel("Price")
    plt.ylabel("Units Sold")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
