# Exploratory Data Analysis (EDA) — Flight Price Prediction

## 1. Project Overview

The objective of this analysis is to understand the flight pricing dataset, identify important pricing patterns, investigate relationships between features, and assess data quality before developing machine learning models.

**Dataset:** Flight Dataset (EaseMyTrip)

**Source file:** `data/raw/Flight_Dataset.xlsx`

**Target variable:** `price`

**Problem type:** Supervised Machine Learning — Regression

### Dataset Summary

| Metric | Value |
|---|---:|
| Total records | 300,153 |
| Total columns | 11 |
| Numerical features | 3 |
| Categorical features | 8 |
| Missing values | 0 |
| Duplicate records | 0 |

The dataset contains flight information covering six airlines and six Indian cities.

---

## 2. Dataset Features

The dataset contains the following columns:

| Feature | Description | Data Type |
|---|---|---|
| airline | Airline operating the flight | Categorical |
| flight | Flight identification number | Categorical |
| source_city | Departure city | Categorical |
| departure_time | Departure time category | Categorical |
| stops | Number of stops | Categorical |
| arrival_time | Arrival time category | Categorical |
| destination_city | Arrival city | Categorical |
| class | Travel class (Economy/Business) | Categorical |
| duration | Total flight duration in hours | Numerical |
| days_left | Days remaining before departure | Numerical |
| price | Flight ticket price in INR | Numerical (Target) |

### Categorical Feature Distribution

| Feature | Unique Values |
|---|---:|
| Airline | 6 |
| Flight | 1,561 |
| Source City | 6 |
| Departure Time | 6 |
| Stops | 3 |
| Arrival Time | 6 |
| Destination City | 6 |
| Class | 2 |

### Numerical Feature Summary

| Statistic | Duration | Days Left | Price |
|---|---:|---:|---:|
| Count | 300,153 | 300,153 | 300,153 |
| Mean | 12.22 | 26.00 | 20,889.66 |
| Standard Deviation | 7.19 | 13.56 | 22,697.77 |
| Minimum | 0.83 | 1 | 1,105 |
| 25th Percentile | 6.83 | 15 | 4,783 |
| Median | 11.25 | 26 | 7,425 |
| 75th Percentile | 16.17 | 38 | 42,521 |
| Maximum | 49.83 | 49 | 123,071 |

---

## 3. Target Variable Analysis — Price

The target variable `price` represents the flight ticket price in Indian Rupees.

### Statistical Observations

- Mean ticket price: ₹20,889.66
- Median ticket price: ₹7,425
- Standard deviation: ₹22,697.77
- Minimum ticket price: ₹1,105
- Maximum ticket price: ₹123,071

### Distribution Analysis

The original price distribution is strongly right-skewed.

The mean is significantly higher than the median, indicating that a relatively small proportion of high-priced tickets contributes substantially to the overall average.

A log transformation using `log1p(price)` was also investigated.

The transformed distribution remains multimodal rather than becoming normally distributed.

This behaviour is likely influenced by the presence of distinct travel-class pricing populations, particularly Economy and Business.

**Finding:**

Flight ticket prices exhibit a strongly right-skewed and multimodal distribution.

The suitability of target transformation will be evaluated during model development rather than applied automatically.

---

## 4. Airline Pricing Analysis

The average ticket price was compared across all six airlines.

### Overall Average Ticket Price

| Airline | Average Price (INR) |
|---|---:|
| Vistara | ~30,400 |
| Air India | ~23,500 |
| SpiceJet | ~6,179 |
| GO FIRST | ~5,652 |
| IndiGo | ~5,324 |
| AirAsia | ~4,091 |

Vistara and Air India exhibit considerably higher overall average ticket prices.

However, this comparison is influenced by differences in travel-class composition.

### Airline Pricing by Travel Class

Further investigation revealed that only Vistara and Air India have Business-class records in this dataset.

| Airline | Class | Mean Price (INR) | Median Price (INR) | Records |
|---|---|---:|---:|---:|
| Vistara | Business | 55,477.03 | 56,588 | 60,589 |
| Air India | Business | 47,131.04 | 49,613 | 32,898 |
| Vistara | Economy | 7,806.94 | 6,461 | 67,270 |
| Air India | Economy | 7,313.68 | 6,082 | 47,994 |
| SpiceJet | Economy | 6,179.28 | 5,654 | 9,011 |
| GO FIRST | Economy | 5,652.01 | 5,336 | 23,173 |
| IndiGo | Economy | 5,324.22 | 4,453 | 43,120 |
| AirAsia | Economy | 4,091.07 | 3,276 | 16,098 |

### Key Findings

1. Only Vistara and Air India operate Business-class services within this dataset.
2. Vistara has the highest average ticket price in both Economy and Business classes.
3. Air India has the second-highest average ticket price in both classes.
4. Airline pricing differences remain visible even when comparing Economy-class records separately.

The Economy-only comparison helps reduce the influence of travel-class composition when examining airline pricing patterns.

---

## 5. Travel Class Analysis

The dataset contains two travel classes:

| Class | Number of Records |
|---|---:|
| Economy | 206,666 |
| Business | 93,487 |

Business-class ticket prices are substantially higher than Economy-class ticket prices.

The boxplot analysis also indicates a wider distribution of prices within Business class.

**Finding:**

Travel class is a major differentiating factor in flight ticket pricing and is expected to be an important predictive feature.

---

## 6. Number of Stops Analysis

The dataset contains three stop categories:

- Zero stops
- One stop
- Two or more stops

The boxplot comparison indicates different price distributions across these categories.

One-stop flights exhibit a particularly wide price distribution.

Potential high-price observations are visible across the stop categories.

**Finding:**

The number of stops may contribute to ticket price variation. Its predictive contribution will be evaluated during model training.

---

## 7. Booking Lead-Time Analysis — Days Left vs Price

The relationship between `days_left` and ticket price was investigated using line plots.

### Overall Observations

Ticket prices generally increase as the departure date approaches.

The average ticket price is considerably higher for bookings made close to departure.

The pricing trend is nonlinear, with a relatively sharp change during the initial booking lead-time period followed by a comparatively stable pattern.

### Travel-Class Comparison

**Economy Class:**

- Prices are higher when booking close to departure.
- Average prices decrease as the booking lead time increases.
- The pricing pattern becomes relatively stable after approximately 20 days.

**Business Class:**

- Business-class prices are considerably higher than Economy prices.
- Prices also show a decline as booking lead time increases.
- The variation is less pronounced compared with Economy class.

### Airline-Level Observations

The airline pricing analysis indicates:

| Booking Lead Time | Vistara | Air India | Other Airlines |
|---|---:|---:|---:|
| 1–2 days | ~₹40,000 | ~₹30,000 | ₹10,000–15,000 |
| 5–7 days | ~₹33,000 | ~₹27,000 | ₹6,000–8,000 |
| 10–15 days | ~₹32,000 | ~₹26,000 | ₹5,000–7,000 |
| 20–25 days | ~₹27,000–30,000 | ~₹20,000–26,000 | ₹4,000–8,000 |

These are approximate observations from the plotted average-price trends, not exact measurements.

An additional Economy-only comparison showed that Vistara and Air India maintained slightly higher average prices than the other airlines across booking lead times.

**Finding:**

The `days_left` feature exhibits a nonlinear relationship with ticket price and is expected to be an important predictive variable.

---

## 8. Flight Duration Analysis

The relationship between flight duration and ticket price was investigated using a scatter plot.

### Observations

- The dataset contains flight durations ranging from approximately 0.83 to 49.83 hours.
- Ticket prices show considerable variation across flight durations.
- Business and Economy classes exhibit distinctly different price distributions.
- Duration alone does not explain the overall ticket price variation.

**Finding:**

Flight duration is a potentially useful numerical predictor. Its contribution should be evaluated alongside airline, route, travel class and other categorical features.

---

## 9. Correlation Analysis

Pearson correlation was calculated between the numerical features.

### Correlation Matrix

| Feature | Duration | Days Left | Price |
|---|---:|---:|---:|
| Duration | 1.0000 | -0.0392 | 0.2042 |
| Days Left | -0.0392 | 1.0000 | -0.0919 |
| Price | 0.2042 | -0.0919 | 1.0000 |

### Key Observations

- Duration has a positive correlation of approximately 0.2042 with price.
- Days left has a negative correlation of approximately -0.0919 with price.
- Duration and days left have a very weak negative correlation.

The relatively weak linear correlations do not necessarily indicate low predictive value.

The visual analysis demonstrates nonlinear relationships, particularly between booking lead time and ticket price.

Tree-based models may capture these nonlinear relationships more effectively than simple linear models.

---

## 10. Route Pricing Analysis

Average ticket prices were calculated for directional source-destination combinations.

The dataset contains 30 possible directional routes between six cities, excluding same-city combinations.

### Highest Average-Priced Routes

| Source City | Destination City | Mean Price (INR) | Median Price (INR) | Records |
|---|---|---:|---:|---:|
| Chennai | Bangalore | 25,081.85 | 10,469 | 6,493 |
| Kolkata | Chennai | 23,660.36 | 8,589 | 6,653 |
| Bangalore | Kolkata | 23,500.06 | 8,112 | 10,028 |
| Bangalore | Chennai | 23,321.85 | 9,241 | 6,410 |
| Mumbai | Bangalore | 23,147.87 | 7,192 | 12,885 |
| Bangalore | Mumbai | 23,128.62 | 7,113 | 12,939 |
| Mumbai | Chennai | 22,781.90 | 8,148 | 10,130 |
| Chennai | Mumbai | 22,765.85 | 8,233 | 9,338 |
| Kolkata | Bangalore | 22,744.81 | 8,111 | 9,824 |
| Chennai | Kolkata | 22,669.93 | 8,394 | 6,983 |

### Key Findings

1. Chennai to Bangalore has the highest average ticket price among the investigated routes.
2. Kolkata to Chennai and Bangalore to Kolkata also exhibit relatively high average prices.
3. Opposite-direction routes can have different average prices.
4. Route-level pricing differences indicate that source and destination information may contribute to prediction.

The route averages include both Economy and Business records. Therefore, they should not be interpreted as isolated route effects.

A combined directional route feature may be investigated during feature engineering.

---

## 11. Data Quality Assessment

The dataset was evaluated for missing values, duplicate records, invalid numerical values and categorical inconsistencies.

### Missing Values

No missing values were identified across the 11 columns.

### Duplicate Records

No exact duplicate rows were identified.

### Numerical Validity

| Validation Check | Invalid Records |
|---|---:|
| Duration <= 0 | 0 |
| Days Left <= 0 | 0 |
| Price <= 0 | 0 |
| Same Source and Destination | 0 |
| Infinite Numerical Values | 0 |

### Categorical Consistency

The categorical features were inspected for their unique values.

The observed categories were consistent with the expected dataset schema.

No unexpected categories or obvious spelling inconsistencies were identified.

**Finding:**

The dataset passed the initial structural and numerical quality checks.

---

## 12. Outlier Analysis

Potential outliers were investigated using the Interquartile Range (IQR) method.

The analysis was performed separately for Economy and Business classes to avoid combining two substantially different price distributions.

### IQR Results

| Travel Class | Potential Outliers | Percentage |
|---|---:|---:|
| Economy | 13,749 | 6.65% |
| Business | 3,650 | 3.90% |

### Interpretation

Potential outliers are present in both travel classes.

However, high-priced tickets may represent legitimate observations influenced by airline, travel class, route, booking lead time and other factors.

Therefore, the identified observations cannot automatically be considered erroneous.

**Decision:**

No outlier records will be removed at this stage.

Their influence will be evaluated during model development and performance analysis.

---

## 13. Key EDA Findings

The major findings from the exploratory analysis are summarized below:

1. **Price Distribution:** Ticket prices are strongly right-skewed and multimodal, with considerable differences between Economy and Business classes.

2. **Travel Class:** Travel class is a major pricing differentiator. Business-class records exist only for Vistara and Air India in this dataset.

3. **Booking Lead Time:** Ticket prices generally increase closer to departure, indicating a nonlinear relationship between `days_left` and price.

4. **Airline Pricing:** Vistara and Air India exhibit higher average prices than other airlines, including within Economy-class comparisons.

5. **Route Pricing:** Different source-destination combinations exhibit different average ticket prices, and opposite-direction routes may have different pricing patterns.

6. **Other Features:** Flight duration, number of stops, departure time and arrival time require further predictive evaluation.

7. **Data Quality:** No missing values, duplicate records, invalid numerical observations or infinite values were identified.

8. **Outliers:** Potential outliers were identified but retained because they may represent legitimate ticket prices.

---

## 14. Preliminary Feature Selection

Based on the exploratory findings, the following features will be considered for initial model development.

| Feature | Initial Decision | Reason |
|---|---|---|
| airline | Retain | Airline-specific pricing differences |
| flight | Exclude initially | High cardinality; potential memorization |
| source_city | Retain | Departure location |
| destination_city | Retain | Arrival location |
| departure_time | Retain | Departure timing |
| arrival_time | Retain | Arrival timing |
| stops | Retain | Journey structure |
| class | Retain | Major pricing differentiator |
| duration | Retain | Journey duration |
| days_left | Retain | Booking lead time |
| price | Target | Prediction variable |

A directional route feature combining `source_city` and `destination_city` will also be investigated.

The usefulness of the `flight` feature will be evaluated experimentally rather than permanently excluded based only on cardinality.

---

## 15. Conclusion

The exploratory data analysis provides an initial understanding of the flight pricing dataset and its major characteristics.

The analysis identifies travel class, airline, booking lead time and route as important potential predictors of ticket prices.

The dataset has passed the initial data quality assessment, and no observations have been removed during EDA.

The next stage will focus on feature engineering, train-test splitting, categorical encoding, numerical preprocessing and building a reusable machine learning preprocessing pipeline.

**Next Phase:** Phase 4 — Feature Engineering and Data Preprocessing.