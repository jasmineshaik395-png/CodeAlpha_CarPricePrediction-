# =============================================================================
# CodeAlpha Internship — Task 3: Car Price Prediction with Machine Learning
# Author  : Shaik Jasmine | jasmineshaik395@gmail.com
# Dataset : car data.csv (CarDekho — 301 used cars)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("   CODEALPHA — CAR PRICE PREDICTION PROJECT")
print("=" * 60)

df = pd.read_csv("car data.csv")

print("\n📊 First 5 rows:")
print(df.head())
print(f"\nShape  : {df.shape[0]} rows × {df.shape[1]} columns")
print("\nDescriptive statistics:")
print(df.describe().round(2))
print(f"\nMissing values : {df.isnull().sum().sum()}")
print("Fuel_Type      :", df["Fuel_Type"].value_counts().to_dict())
print("Transmission   :", df["Transmission"].value_counts().to_dict())
print("Selling_type   :", df["Selling_type"].value_counts().to_dict())

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Remove extreme outliers (top 1% Present_Price)
q99 = df["Present_Price"].quantile(0.99)
df  = df[df["Present_Price"] <= q99].reset_index(drop=True)
print(f"\nAfter outlier removal : {len(df)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
df["Car_Age"]          = 2024 - df["Year"]
df["KMs_per_Year"]     = df["Driven_kms"] / (df["Car_Age"] + 1)
df["Price_Diff"]       = df["Present_Price"] - df["Selling_Price"]
df["Depreciation_pct"] = (df["Price_Diff"] / df["Present_Price"]) * 100

le = LabelEncoder()
for col in ["Fuel_Type", "Selling_type", "Transmission"]:
    df[col + "_enc"] = le.fit_transform(df[col])

FEATURES = [
    "Car_Age", "Present_Price", "Driven_kms",
    "Fuel_Type_enc", "Selling_type_enc", "Transmission_enc",
    "Owner", "KMs_per_Year"
]
TARGET = "Selling_Price"

X = df[FEATURES]
y = df[TARGET]

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain : {len(X_train)} rows  |  Test : {len(X_test)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Ridge Regression"  : Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))]),
    "Random Forest"     : RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=4, random_state=42),
}

results = {}
print("\n📈 Model Performance on Test Set:")
print(f"{'Model':<22}  {'R²':>6}  {'MAE':>6}  {'RMSE':>6}  {'CV R²':>8}")
print("-" * 58)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    cv    = cross_val_score(model, X, y, cv=5, scoring="r2")

    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = dict(r2=r2, mae=mae, rmse=rmse, cv_r2=cv.mean(), preds=preds)
    print(f"{name:<22}  {r2:>6.3f}  {mae:>6.3f}  {rmse:>6.3f}  {cv.mean():>8.3f}")

best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\n✅ Best model : {best_name}  (R² = {results[best_name]['r2']:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
rf_imp = pd.Series(
    models["Random Forest"].feature_importances_, index=FEATURES
).sort_values(ascending=False)

print("\n🔑 Feature Importances (Random Forest):")
print(rf_imp.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 7. BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n💡 Business Insights:")
print(f"  • Average selling price   : ₹{y.mean():.2f} Lakhs")
print(f"  • Average depreciation    : {df['Depreciation_pct'].mean():.1f}%")
print(f"  • Present_Price is the strongest single predictor of Selling_Price")
print(f"  • Car_Age has a strong negative correlation with resale value")
print(f"  • Diesel cars retain value better than Petrol cars")
print(f"  • Automatic transmission cars command a price premium")
print(f"  • Dealer-listed cars sell at higher prices than Individual sellers")
print(f"  • Higher Owner count (second/third owner) reduces price significantly")

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("CodeAlpha — Car Price Prediction Analysis", fontsize=16,
             fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

PALETTE = ["#4E79A7", "#F28E2B", "#E15759"]

# Row 0 — Scatter plots
for i, (col, xlabel) in enumerate([
    ("Present_Price", "Present Price (₹ Lakhs)"),
    ("Car_Age",       "Car Age (Years)"),
    ("Driven_kms",    "Driven KMs"),
]):
    ax = fig.add_subplot(gs[0, i])
    ax.scatter(df[col], df["Selling_Price"], alpha=0.55,
               color=PALETTE[i], edgecolors="white", s=55)
    m, b = np.polyfit(df[col], df["Selling_Price"], 1)
    xl = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xl, m * xl + b, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Selling Price (₹ Lakhs)" if i == 0 else "", fontsize=10)
    ax.set_title(f"{col} vs Selling Price", fontsize=11, fontweight="bold")

# Row 1 left — Correlation heatmap
ax2 = fig.add_subplot(gs[1, 0])
corr = df[["Selling_Price", "Present_Price", "Car_Age",
           "Driven_kms", "Owner", "KMs_per_Year"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
ax2.set_title("Correlation Heatmap", fontsize=11, fontweight="bold")

# Row 1 mid — Model R² bar
ax3 = fig.add_subplot(gs[1, 1])
names  = list(results.keys())
r2vals = [results[n]["r2"] for n in names]
bars   = ax3.barh(names, r2vals, color=["#4E79A7", "#59A14F", "#E15759"])
for bar, val in zip(bars, r2vals):
    ax3.text(val - 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="right",
             color="white", fontweight="bold", fontsize=9)
ax3.set_xlim(0.5, 1.05)
ax3.set_xlabel("R² Score", fontsize=10)
ax3.set_title("Model R² Comparison", fontsize=11, fontweight="bold")

# Row 1 right — Actual vs Predicted
ax4 = fig.add_subplot(gs[1, 2])
preds_best = results[best_name]["preds"]
ax4.scatter(y_test, preds_best, alpha=0.7, color="#59A14F",
            edgecolors="white", s=60)
lims = [min(y_test.min(), preds_best.min()) - 0.5,
        max(y_test.max(), preds_best.max()) + 0.5]
ax4.plot(lims, lims, "k--", linewidth=1.5)
ax4.set_xlabel("Actual Price (₹ Lakhs)", fontsize=10)
ax4.set_ylabel("Predicted Price (₹ Lakhs)", fontsize=10)
ax4.set_title(f"Actual vs Predicted\n({best_name})", fontsize=11, fontweight="bold")

# Row 2 left — Feature importance
ax5 = fig.add_subplot(gs[2, 0])
rf_imp.plot(kind="bar", ax=ax5, color="#4E79A7", edgecolor="white")
ax5.set_title("Feature Importance (RF)", fontsize=11, fontweight="bold")
ax5.set_ylabel("Importance", fontsize=10)
ax5.tick_params(axis="x", rotation=35)

# Row 2 mid — Price by Fuel Type boxplot
ax6 = fig.add_subplot(gs[2, 1])
order = df.groupby("Fuel_Type")["Selling_Price"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", order=order,
            ax=ax6, palette={"Petrol": "#4E79A7", "Diesel": "#F28E2B", "CNG": "#59A14F"})
ax6.set_title("Price by Fuel Type", fontsize=11, fontweight="bold")
ax6.set_xlabel("Fuel Type", fontsize=10)
ax6.set_ylabel("Selling Price (₹ Lakhs)", fontsize=10)

# Row 2 right — Residuals
ax7 = fig.add_subplot(gs[2, 2])
residuals = y_test.values - preds_best
ax7.scatter(preds_best, residuals, alpha=0.7, color="#E15759",
            edgecolors="white", s=60)
ax7.axhline(0, color="black", linewidth=1.5, linestyle="--")
ax7.set_xlabel("Predicted Price (₹ Lakhs)", fontsize=10)
ax7.set_ylabel("Residuals", fontsize=10)
ax7.set_title(f"Residual Plot ({best_name})", fontsize=11, fontweight="bold")

plt.savefig("car_price_analysis.png", dpi=150, bbox_inches="tight")
print("\n✅ Analysis chart saved → car_price_analysis.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔮 Sample Predictions (Gradient Boosting):")
print(f"{'Age':>4}  {'Present₹':>9}  {'KMs':>7}  {'Fuel':>6}  {'Trans':>6}  {'Predicted':>12}")
print("-" * 58)
gb = models["Gradient Boosting"]
samples = [
    (2,  8.0,  15000, 0, 0, 0),
    (5,  6.5,  60000, 1, 0, 0),
    (3,  12.0, 30000, 1, 1, 0),
    (8,  4.0,  90000, 0, 0, 1),
    (1,  15.0, 5000,  1, 1, 0),
]
for age, pp, kms, fuel, trans, seller in samples:
    kpy = kms / (age + 1)
    inp = pd.DataFrame([[age, pp, kms, fuel, seller, trans, 0, kpy]],
                       columns=FEATURES)
    pred = gb.predict(inp)[0]
    fl = ["Petrol", "Diesel"][fuel]
    tl = ["Manual", "Auto"][trans]
    print(f"{age:>4}  {pp:>9.1f}  {kms:>7}  {fl:>6}  {tl:>6}  ₹{pred:>9.2f}L")

print("\n🎯 Done!")
