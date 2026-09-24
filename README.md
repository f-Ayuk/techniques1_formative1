# trial_formative-1

# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## Setup

```bash
git clone <repo>
cd formative1
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt

## Data
Download the Milan dataset from Harvard Dataverse:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV


# Milan Mobile Network Traffic Forecasting

One-step-ahead Internet-traffic forecasting on the Telecom Italia
Milan dataset (10,000 areas, 10-minute intervals, Nov–Dec 2013).

## Models
- **SARIMA** — classical statistical baseline (seasonal ARMA, s=144)
- **LSTM**  — deep recurrent neural network
- **CNN**   — deep dilated causal convolutional network

## Setup
1. Place all `sms-call-internet-mi-*.txt` files in `milan_data/`.
2. `pip install -r requirements.txt`
3. Open `notebooks/milan_traffic_forecasting.ipynb` and run all cells.

## Outputs
- `figures/eda/` — EDA plots
- `figures/forecasts/` — 9 superposed prediction plots
- `results/` — metrics, timings, predictions, experiment log