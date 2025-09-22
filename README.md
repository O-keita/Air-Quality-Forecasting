# Air-Quality-Forecasting


This project aims to predict PM2.5 concentrations in Beijing using historical air quality and weather data. By leveraging Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models, this assignment provides hands-on experience in preprocessing sequential data, designing and optimizing time series models, and evaluating their performance using RMSE.
The objective is to develop a robust predictive model that accurately forecasts PM2.5 levels and demonstrates the practical application of machine learning techniques in environmental monitoring and public health protection.





## Data
 - Beijing PM2.5 Forecasting dara
 - Target **PM2.5 concentration**
 - Preprocessing:
     - Outlier detection
     - MinMax Scaling (fit on training, applied  to test)
     - Sliding windows sequence creation
  

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

**clone Repo**
```bash
https://github.com/O-keita/Air-Quality-Forecasting.git
```



