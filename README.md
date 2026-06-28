# Hotel Booking Cancellation Prediction

Predicts whether a hotel booking will be **cancelled** — trained on ~119k real
bookings — and serves it through an interactive **Streamlit dashboard** with a
live predictor and exploratory analytics.

**Model:** XGBoost · **Accuracy ≈ 87%** · **ROC-AUC ≈ 0.944** (held-out test)

## Why it matters

Cancellations cost hotels revenue and break inventory planning. A reliable
cancellation-risk score lets revenue teams overbook intelligently, target
retention offers, and forecast occupancy — turning a historical dataset into an
operational decision tool.

## What's inside

```
train_model.py     reproducible training pipeline -> model.pkl + encoders
app.py             Streamlit app: live predictor + EDA dashboard (Plotly)
Hotel_booking.ipynb  exploratory analysis notebook
hotel_bookings.csv / .parquet   dataset
*.pkl              saved model, feature columns, label encoders
```

## Modelling highlights (the parts that matter)

- **Leakage control** — drops `reservation_status` / `reservation_status_date`
  (which directly encode the target) and other post-hoc fields, so the reported
  accuracy is honest and not inflated by leakage.
- **Data cleaning** — removes impossible "zero-guest" bookings, imputes missing
  `children`/`country`.
- **Feature engineering** — `total_nights`, `total_guests`, `is_family`,
  `room_changed`, `is_weekend_only`, `revenue`, `lead_time_log`.
- **Model** — `XGBClassifier` with cross-validated evaluation (accuracy,
  ROC-AUC, confusion matrix, classification report).

## The app

A multi-page Streamlit dashboard:
- **Predictor** — enter a booking's details, get its cancellation probability.
- **EDA** — interactive Plotly charts of cancellation drivers (lead time, deposit
  type, market segment, etc.).

## Run it

```bash
pip install -r requirements.txt
python train_model.py        # (re)generates model.pkl + encoders
streamlit run app.py         # launches the dashboard
```

## Deploy

The app is ready for **Streamlit Community Cloud**: point it at this repo with
`app.py` as the entry file (model artifacts are committed, so no training needed
at deploy time).

## Tech stack

Python · pandas · **XGBoost** · scikit-learn · **Streamlit** · Plotly · joblib
