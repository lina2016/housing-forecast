import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ---------------- CONFIG ----------------
CSV_PATH = "data/zillow_county.csv"  # put Zillow county-level CSV here
LAGS = 12
FORECAST_MONTHS = 6
EPOCHS = 50
BATCH_SIZE = 16

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    # Extract date columns (YYYY-MM format)
    date_cols = [c for c in df.columns if c[:4].isdigit()]
    df_long = df.melt(
        id_vars=["RegionName", "StateName"],
        value_vars=date_cols,
        var_name="date", value_name="ZHVI"
    )
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long.sort_values(["StateName",  "date"], inplace=True)
    return df_long

df_long = load_data()

# ---------------- MODEL FUNCTION ----------------
def forecast_series(ts):
    ts = ts.dropna()
    if len(ts) < LAGS + FORECAST_MONTHS:
        return None, None  # not enough data

    # Create lag features
    data = pd.DataFrame({f"lag_{i}": ts.shift(i) for i in range(LAGS, 0, -1)})
    data["target"] = ts.values
    data.dropna(inplace=True)

    X = data.drop(columns=["target"]).values
    y = data["target"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler().fit(X)
    scaler_y = MinMaxScaler().fit(y)
    Xs, ys = scaler_X.transform(X), scaler_y.transform(y)

    split = int(len(Xs) * 0.8)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(LAGS,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    with st.spinner("Training model... Please wait ⏳"):
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # Forecast future
    last_win = Xs[-1].reshape(1, -1)
    preds = []
    for _ in range(FORECAST_MONTHS):
        next_s = model.predict(last_win, verbose=0)[0][0]
        preds.append(scaler_y.inverse_transform([[next_s]])[0][0])
        last_win = np.roll(last_win, -1)
        last_win[0, -1] = next_s

    future_dates = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=FORECAST_MONTHS, freq="MS")
    return future_dates, preds

# ---------------- UI ----------------
st.title("🏠 Housing Price Forecast Explorer")

# --- State dropdown ---
states = sorted(df_long["StateName"].dropna().unique())
default_state = "CA"

state = st.selectbox(
    "Select a State",
    states,
    index=states.index(default_state) if default_state in states else 0
)

# --- County dropdown ---
counties = sorted(df_long[df_long["StateName"] == state]["RegionName"].dropna().unique())
default_county = "Santa Clara County"

county = st.selectbox(
    "Select a County",
    counties,
    index=counties.index(default_county) if default_county in counties else 0
)


# Filter selected county data
df_sel = df_long[(df_long["StateName"] == state) & (df_long["RegionName"] == county)]
ts = df_sel.set_index("date")["ZHVI"]

# ---------------- CHART 1: HISTORICAL ----------------
st.subheader(f"📈 Historical Prices for {county}, {state}")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(ts.index, ts.values, label="Historical Prices")
ax1.set_ylabel("ZHVI ($)")
ax1.set_title(f"{county}, {state} - Historical Housing Prices")
ax1.legend()
plt.xticks(rotation=45)
st.pyplot(fig1)

# ---------------- CHART 2: FORECAST ----------------
future_dates, preds = forecast_series(ts)

if preds is not None:
    st.subheader("🔮 6-Month Forecast")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(future_dates, preds, "o--", color="red", label="Forecast")
    ax2.set_ylabel("ZHVI ($)")
    ax2.set_title(f"{county}, {state} - Next 6 Months Forecast")
    ax2.legend()

    # Format X axis as "Mon-YYYY"
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.success("✅ Forecast ready!")
    st.write("**Next 6 months prediction values:**")
    for d, v in zip(future_dates, preds):
        st.write(f"{d.strftime('%B-%Y')} : ${v:,.0f}")
else:
    st.error("Not enough data to forecast this county.")
