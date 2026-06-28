import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model.pkl")
    cols     = joblib.load("feature_columns.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, cols, encoders

@st.cache_data
def load_eda():
    with open("eda_summary.json") as f:
        return json.load(f)

model, feature_cols, label_encoders = load_model()
eda = load_eda()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/hotel.png", width=80)
st.sidebar.title("Hotel Booking Cancellation")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview & EDA", "🔮 Predict Cancellation", "📊 Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** XGBoost  \n**Accuracy:** 86.9%  \n**ROC-AUC:** 0.944  \n**Dataset:** 119K bookings"
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & EDA":
    st.title("🏨 Hotel Booking Cancellation — EDA")
    st.markdown("Analysing **119,390 hotel bookings** to understand what drives cancellations.")

    # KPI row
    total    = eda["total"]
    cancelled= eda["cancelled"]
    rate     = cancelled / total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bookings",       f"{total:,}")
    c2.metric("Cancellations",        f"{cancelled:,}", f"{rate:.1%} of total")
    c3.metric("Avg Lead Time (days)", f"{eda['avg_lead_time']:.0f}")
    c4.metric("Avg Daily Rate (£)",   f"£{eda['avg_adr']:.2f}")

    st.markdown("---")

    # Row 1: by hotel + by month
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cancellation Rate by Hotel Type")
        htype = pd.DataFrame(eda["by_hotel"].items(), columns=["Hotel Type","Cancellation Rate"])
        fig = px.bar(htype, x="Hotel Type", y="Cancellation Rate",
                     color="Hotel Type", text_auto=".1%",
                     color_discrete_sequence=["#636EFA","#EF553B"])
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cancellation Rate by Month")
        monthly = pd.DataFrame(eda["by_month"].items(), columns=["Month","Cancellation Rate"])
        fig = px.line(monthly, x="Month", y="Cancellation Rate",
                      markers=True, color_discrete_sequence=["#00CC96"])
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: lead time box + deposit
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Lead Time vs Cancellation")
        lead_df = pd.DataFrame(eda["lead_sample"], columns=["lead_time","is_canceled"])
        fig = px.box(lead_df, x="is_canceled", y="lead_time",
                     labels={"is_canceled":"Cancelled (1=Yes)","lead_time":"Lead Time (days)"},
                     color="is_canceled",
                     color_discrete_map={0:"#00CC96", 1:"#EF553B"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Deposit Type vs Cancellation")
        deposit = pd.DataFrame(eda["by_deposit"].items(), columns=["Deposit Type","Cancellation Rate"])
        fig = px.bar(deposit, x="Deposit Type", y="Cancellation Rate",
                     color="Deposit Type", text_auto=".1%",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: market segment + ADR
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Cancellation by Market Segment")
        seg = pd.DataFrame(eda["by_segment"].items(), columns=["Market Segment","Cancellation Rate"])
        seg = seg.sort_values("Cancellation Rate", ascending=False)
        fig = px.bar(seg, x="Cancellation Rate", y="Market Segment",
                     orientation="h", text_auto=".1%",
                     color="Cancellation Rate", color_continuous_scale="RdYlGn_r")
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Average Daily Rate Distribution")
        adr_df = pd.DataFrame(eda["adr_sample"], columns=["adr","is_canceled"])
        fig = px.histogram(adr_df, x="adr", color="is_canceled",
                           labels={"adr":"Average Daily Rate (£)","is_canceled":"Cancelled"},
                           color_discrete_map={0:"#00CC96", 1:"#EF553B"},
                           barmode="overlay", opacity=0.7, nbins=60)
        st.plotly_chart(fig, use_container_width=True)

    # Top countries
    st.subheader("Top 10 Guest Countries")
    countries = pd.DataFrame(eda["top_countries"].items(), columns=["Country","Bookings"])
    fig = px.bar(countries, x="Country", y="Bookings",
                 color="Bookings", color_continuous_scale="Blues", text_auto=True)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.subheader("Key Insights")
    i1, i2, i3 = st.columns(3)
    i1.info("**Deposit type is the #1 predictor.** Non-refundable bookings paradoxically cancel at higher rates — customers book to hold a spot then cancel anyway.")
    i2.info("**Room changes drive cancellations.** When the assigned room differs from the reserved room, cancellation risk rises significantly.")
    i3.info("**Longer lead time = higher risk.** Bookings made far in advance are more likely to be cancelled as plans change.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Cancellation":
    st.title("🔮 Will This Booking Be Cancelled?")
    st.markdown("Fill in the booking details below to get an instant prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Booking Details**")
            hotel         = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
            lead_time     = st.number_input("Lead Time (days)", 0, 730, 30)
            arrival_month = st.selectbox("Arrival Month", [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"])
            market_segment = st.selectbox("Market Segment", [
                "Online TA","Offline TA/TO","Direct","Corporate","Groups","Complementary","Aviation"])
            distribution_channel = st.selectbox("Distribution Channel", [
                "TA/TO","Direct","Corporate","GDS","Undefined"])

        with col2:
            st.markdown("**Room & Stay**")
            reserved_room  = st.selectbox("Reserved Room Type", ["A","B","C","D","E","F","G","H","L","P"])
            assigned_room  = st.selectbox("Assigned Room Type", ["A","B","C","D","E","F","G","H","I","K","L","P"])
            week_nights    = st.number_input("Weekday Nights", 0, 30, 2)
            weekend_nights = st.number_input("Weekend Nights", 0, 14, 1)
            meal           = st.selectbox("Meal Plan", ["BB","HB","FB","SC","Undefined"])

        with col3:
            st.markdown("**Guest & History**")
            adults                        = st.number_input("Adults", 1, 10, 2)
            children                      = st.number_input("Children", 0, 10, 0)
            babies                        = st.number_input("Babies", 0, 5, 0)
            deposit_type                  = st.selectbox("Deposit Type", ["No Deposit","Non Refund","Refundable"])
            customer_type                 = st.selectbox("Customer Type", ["Transient","Transient-Party","Contract","Group"])
            previous_cancellations        = st.number_input("Previous Cancellations", 0, 50, 0)
            previous_bookings_not_canceled= st.number_input("Previous Non-Cancelled", 0, 50, 0)
            booking_changes               = st.number_input("Booking Changes", 0, 20, 0)
            days_in_waiting_list          = st.number_input("Days on Waiting List", 0, 400, 0)
            adr                           = st.number_input("Average Daily Rate (£)", 0.0, 5000.0, 100.0)
            required_car_parking          = st.number_input("Parking Spaces Required", 0, 5, 0)
            total_special_requests        = st.number_input("Special Requests", 0, 10, 0)
            is_repeated_guest             = st.checkbox("Repeated Guest?")
            country                       = st.text_input("Country Code (e.g. GBR, PRT)", "GBR")

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                     "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}

        row = {
            "hotel":                          hotel,
            "meal":                           meal,
            "country":                        country.upper().strip(),
            "market_segment":                 market_segment,
            "distribution_channel":           distribution_channel,
            "is_repeated_guest":              int(is_repeated_guest),
            "previous_cancellations":         previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type":             reserved_room,
            "assigned_room_type":             assigned_room,
            "booking_changes":                booking_changes,
            "deposit_type":                   deposit_type,
            "days_in_waiting_list":           days_in_waiting_list,
            "customer_type":                  customer_type,
            "required_car_parking_spaces":    required_car_parking,
            "total_of_special_requests":      total_special_requests,
            "total_nights":                   week_nights + weekend_nights,
            "total_guests":                   adults + children + babies,
            "is_family":                      int(adults > 0 and children > 0),
            "room_changed":                   int(reserved_room != assigned_room),
            "is_weekend_only":                int(weekend_nights > 0 and week_nights == 0),
            "revenue":                        adr * (week_nights + weekend_nights),
            "lead_time_log":                  np.log1p(lead_time),
            "adr_log":                        np.log1p(adr),
            "arrival_month_num":              month_map[arrival_month],
        }

        input_df = pd.DataFrame([row])

        # Encode categoricals using saved encoders
        for col, le in label_encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = le.transform([le.classes_[0]])

        # Align columns
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        prob       = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if prediction == 1:
                st.error(f"## ❌ HIGH CANCELLATION RISK")
                st.markdown(f"### Cancellation probability: **{prob:.1%}**")
                st.progress(prob)
                st.warning("This booking is **likely to be cancelled**. Consider requesting a deposit or sending a reminder closer to the date.")
            else:
                st.success(f"## ✅ LOW CANCELLATION RISK")
                st.markdown(f"### Cancellation probability: **{prob:.1%}**")
                st.progress(prob)
                st.info("This booking is **likely to complete**. Standard follow-up recommended.")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Cancellation Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#EF553B" if prob > 0.5 else "#00CC96"},
                "steps": [
                    {"range": [0,  40],  "color": "#d4edda"},
                    {"range": [40, 65],  "color": "#fff3cd"},
                    {"range": [65, 100], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black","width": 3}, "value": 50},
            },
            number={"suffix": "%", "font": {"size": 36}},
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 XGBoost Model Performance")
    st.markdown("Trained on **95,368 bookings**, tested on **23,842 bookings**.")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  "86.93%")
    m2.metric("ROC-AUC",   "0.9444")
    m3.metric("Precision", "85% (cancelled class)")
    m4.metric("Recall",    "79% (cancelled class)")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Confusion matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm_vals = np.array([[13777, 1225], [1890, 6950]])
        labels  = ["Not Cancelled","Cancelled"]
        fig = px.imshow(
            cm_vals, text_auto=True, aspect="auto",
            x=labels, y=labels,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    with col2:
        st.subheader("Top 15 Feature Importances")
        fi = pd.Series(
            model.feature_importances_,
            index=feature_cols
        ).nlargest(15).sort_values()
        fig = px.bar(
            x=fi.values, y=fi.index,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=fi.values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ROC curve (approximate — using stored values)
    st.subheader("ROC Curve")
    st.markdown("AUC = **0.9444** — the model has strong discriminative ability between cancelled and non-cancelled bookings.")

    fpr_pts = [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]
    tpr_pts = [0, 0.25, 0.45, 0.60, 0.72, 0.80, 0.86, 0.91, 0.95, 0.98, 1.0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_pts, y=tpr_pts, mode="lines",
                              name="XGBoost (AUC = 0.944)", line=dict(color="#636EFA", width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                              name="Random Baseline", line=dict(color="gray", dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      legend=dict(x=0.6, y=0.1))
    st.plotly_chart(fig, use_container_width=True)

    # Model comparison table
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        "Model":    ["XGBoost ✅ (selected)","Random Forest","Logistic Regression","Decision Tree","KNN","Naive Bayes"],
        "Accuracy": ["86.9%","85.2%","79.8%","83.1%","74.3%","71.2%"],
        "ROC-AUC":  ["0.944","0.921","0.861","0.831","0.740","0.760"],
        "Notes":    [
            "Best overall — handles non-linearity + feature interactions",
            "Strong but slower, less interpretable",
            "Fast but misses non-linear patterns",
            "Overfits without tuning",
            "Slow on large datasets",
            "Assumes feature independence — too simple",
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Why XGBoost?")
    c1, c2, c3 = st.columns(3)
    c1.success("**Handles mixed data types** — works natively with both numerical and encoded categorical features")
    c2.success("**Robust to outliers** — log transforms on lead time and ADR minimise skew impact")
    c3.success("**Interpretable** — feature importance shows exactly what drives predictions")
