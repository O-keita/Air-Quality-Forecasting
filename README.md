# Air-Quality-Forecasting

This project aims to predict PM2.5 concentrations in Beijing using historical air quality and weather data. By leveraging Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models, this assignment provides hands-on experience in preprocessing sequential data, designing and optimizing time series models, and evaluating their performance using RMSE.  
The objective is to develop a robust predictive model that accurately forecasts PM2.5 levels and demonstrates the practical application of machine learning techniques in environmental monitoring and public health protection.

---

## Data

- **Beijing PM2.5 Forecasting Data**
- **Target:** PM2.5 concentration
- **Preprocessing Steps:**
  - Loading the train and test datasets as pandas DataFrames
  - Handling missing values (filled with mean)
  - Extracting time-based features: hour, day, month, weekend
  - Outlier detection (IQR method on PM2.5)
  - MinMax scaling (fit on training, applied to test)
  - Correlation analysis (heatmap)
  - Barplot for outlier visualization
  - Sliding window sequence creation for LSTM input

---

## Model Architecture

```text
Input (24 timesteps × 12 features)
↓
Bidirectional LSTM (128 units) → BatchNorm → Dropout(0.3)
↓
Bidirectional LSTM (64 units) → BatchNorm → Dropout(0.2)
↓
Bidirectional LSTM (32 units) → Dropout(0.2)
↓
Bidirectional LSTM (16 units, ReLU)
↓
Dense(1) → PM2.5 prediction
```

- **Optimizer:** Adam (tunable learning rate)
- **Loss:** Mean Squared Error (MSE)
- **Metric:** RMSE (custom lambda)
- **Activations:** tanh and ReLU
- **Regularization:** Dropout (0.2–0.3)
- **Batch size:** 32 or 64 (experimented)
- **Epochs:** 10–15 (experimented)

---

## Experimentation

A series of experiments were conducted with different model configurations, hyperparameters, and preprocessing strategies.  
Key aspects of experimentation included:

- Varying LSTM stack depth and units
- Trying both LSTM and Bidirectional LSTM variants
- Adjusting dropout, batch size, and learning rate
- Comparing ReLU and tanh activations
- Applying different scalers (StandardScaler, MinMaxScaler)
- Detecting and removing outliers
- Engineering time-based features

**Example experiment table:**

| Exp | Model Type | Layers (Units)         | Activation | Dropout | Optimizer | Batch Size | Epochs | Scaled | RMSE    |
|-----|------------|------------------------|-----------|---------|-----------|------------|--------|--------|---------|
| 1   | LSTM       | 64                     | ReLU      | 0.2     | Adam      | 32         | 10     | No     | 4000    |
| 10  | LSTM       | 128, 64, 32            | Tanh      | 0.3     | Adam      | 32         | 10     | No     | 3700    |
| 14  | Bi-LSTM    | 64                     | ReLU      | 0.2     | Adam      | 32         | 10     | Yes    | 0.182   |
| 20  | Bi-LSTM    | 128, 64, 32, 16        | Tanh/ReLU | 0.3     | Adam      | 64         | 15     | No     | 0.182   |

---

## Results & Discussion

- **Metric:** Root Mean Squared Error (RMSE)
    - RMSE is the square root of the mean of squared errors between predictions and true PM2.5 values.
- **Findings:**
    - Scaling and outlier removal improve model stability, but not always RMSE.
    - Deeper and bidirectional architectures generally outperform shallow ones, up to a point.
    - Dropout helps prevent overfitting.
    - The model has difficulty predicting sudden spikes in PM2.5 due to external, unmodeled factors.
- **Challenges:**
    - Vanishing gradients in deep LSTM stacks; mitigated with tanh activations and batch normalization.
    - Overfitting with too many layers; solved with dropout and early stopping.

---

## Usage

**Clone the Repo**
```bash
git clone https://github.com/O-keita/Air-Quality-Forecasting.git
cd Air-Quality-Forecasting
```

**Install requirements**
```bash
pip install -r requirements.txt
```

**Prepare Data**
1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the [Kaggle competition](https://www.kaggle.com/).
2. Place files in the `data/` directory.

**Run the Notebook**
- Open and run `air_quality_forecasting_starter_code.ipynb` in Jupyter Notebook or Colab.
- Follow the notebook to preprocess data, train, validate, and generate predictions.

**Submit**
- Format your predictions according to `sample_submission.csv` and submit on Kaggle.

---

## Project Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── air_quality_forecasting_starter_code.ipynb
├── requirements.txt
├── README.md

```

---

## Reference

- [Statistics by Jim: Root Mean Squared Error (RMSE)](https://statisticsbyjim.com/regression/root-mean-square-error-rmse/#google_vignette)

---

## Author

- **Omar Keita**
- [GitHub Repository](https://github.com/O-keita/Air-Quality-Forecasting)

---
