# 🚗 Car Price Prediction with Machine Learning
### CodeAlpha Data Science Internship — Task 3

---

## 📌 Project Overview

This project predicts the **resale price of used cars** based on features like brand, present price, car age, fuel type, transmission, and mileage using regression and ensemble machine learning models.

The goal is to help buyers and sellers make informed decisions by estimating fair market prices for used cars.

---

## 📂 Repository Structure

```
CodeAlpha_CarPricePrediction/
│
├── car data.csv                  # Dataset (301 used cars — CarDekho)
├── car_price_prediction.py       # Main Python script
├── car_price_analysis.png        # Output visualisation (auto-generated)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 📊 Dataset Description

| Column | Description |
|---|---|
| `Car_Name` | Name/model of the car |
| `Year` | Year of manufacture |
| `Selling_Price` | Resale price (₹ Lakhs) — **target** |
| `Present_Price` | Current showroom price (₹ Lakhs) |
| `Driven_kms` | Total kilometres driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Selling_type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

- **301 rows**, **no missing values**
- Source: CarDekho used car listings dataset

---

## 🛠️ Steps Performed

1. **Data Loading & Exploration** — shape, dtypes, statistics, null check
2. **Data Cleaning** — duplicates removed, outlier handling (top 1% Present_Price)
3. **Feature Engineering** — Car_Age, KMs_per_Year, Depreciation_pct, interaction features
4. **Label Encoding** — Fuel_Type, Selling_type, Transmission → numeric
5. **Model Training** — 3 regression models with 80/20 train-test split
6. **Model Evaluation** — R², MAE, RMSE, 5-fold cross-validation
7. **Visualisation** — 7-panel analysis chart saved as PNG
8. **Business Insights** — actionable findings for buyers & sellers

---

## 🤖 Models Used

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Ridge Regression | 0.706 | 1.447 | 2.447 |
| Random Forest | 0.948 | 0.626 | 1.026 |
| **Gradient Boosting** ✅ | **0.955** | **0.632** | **0.956** |

✅ **Best Model: Gradient Boosting** (R² = 0.955)

---

## 🔑 Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Present_Price | 0.864 |
| Car_Age | 0.081 |
| Driven_kms | 0.024 |
| KMs_per_Year | 0.010 |
| Selling_type | 0.009 |

---

## 💡 Key Business Insights

- **Present_Price** is the strongest predictor of resale value (86.4% importance)
- **Car_Age** has a strong negative impact — older cars sell for significantly less
- **Diesel** cars retain value better than Petrol cars
- **Automatic** transmission commands a price premium over Manual
- **Dealer** listings fetch higher prices than Individual sellers
- **Second/third owners** see a significant drop in resale value
- Average depreciation across the dataset is **36.7%**

---

## 🔮 Sample Predictions

| Age | Present Price | KMs | Fuel | Transmission | Predicted Price |
|---|---|---|---|---|---|
| 2 yrs | ₹8.0L | 15,000 | Petrol | Manual | ₹7.09L |
| 5 yrs | ₹6.5L | 60,000 | Diesel | Manual | ₹4.40L |
| 3 yrs | ₹12.0L | 30,000 | Diesel | Auto | ₹8.34L |
| 8 yrs | ₹4.0L | 90,000 | Petrol | Manual | ₹2.36L |
| 1 yr | ₹15.0L | 5,000 | Diesel | Auto | ₹13.09L |

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ShaikJasmine/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python car_price_prediction.py
```

The script prints model metrics to the console and saves **`car_price_analysis.png`** to the current directory.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| **Pandas** | Data loading, cleaning, feature engineering |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Model training, evaluation, pipelines |
| **Matplotlib** | Charts and dashboards |
| **Seaborn** | Heatmaps and boxplots |

---

## 🔧 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🙏 Acknowledgements

- **CodeAlpha** for the internship opportunity
- Dataset: CarDekho used car listings

---

## 📬 Connect

- 🔗 [LinkedIn](https://www.linkedin.com/in/ShaikJasmine) — *tag @CodeAlpha when you post!*
- 📧 [jasmineshaik395@gmail.com](mailto:jasmineshaik395@gmail.com)
- 💻 [GitHub](https://github.com/ShaikJasmine)
