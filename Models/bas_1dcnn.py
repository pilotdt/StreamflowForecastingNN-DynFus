import time as pytime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from hydroeval import evaluator, nse, kge


folder_with_data = 'Data'
river_flow = pd.read_csv(folder_with_data + '/river-flow.csv')
river_flow['dateTime'] = pd.to_datetime(river_flow['dateTime'], errors='raise')
river_flow['dateTime'] = river_flow['dateTime'].dt.date
river_flow = river_flow.groupby('dateTime').mean().reset_index()

climate_variables = pd.read_csv(folder_with_data + '/atmospheric-variables.csv')
climate_variables['dateTime'] = pd.to_datetime(climate_variables['dateTime'], errors='raise')
climate_variables['dateTime'] = climate_variables['dateTime'].dt.date
climate_variables = climate_variables.set_index('dateTime')

river_flow = river_flow.drop(columns={'Unnamed: 0'})
weather_data = pd.merge(climate_variables, river_flow, on=['dateTime'], how='inner')

weather_data = weather_data.iloc[:,[0,3,1,4,5,6,7,8,9,10,11,12,13]]
time_steps = 30 
horizon = 7 

numerical_columns = weather_data.select_dtypes(include=['float64']).columns
datetime_columns = weather_data.select_dtypes(include=['datetime64']).columns

train_end = int(len(weather_data) * 0.60)
val_end   = int(len(weather_data) * 0.80)

train_df = weather_data.iloc[:train_end].copy()
val_df   = weather_data.iloc[train_end:val_end].copy()
test_df  = weather_data.iloc[val_end:].copy()

train_df = train_df.interpolate(method='linear', limit_direction='both', axis=0)
val_df   = val_df.interpolate(method='linear', limit_direction='both', axis=0)
test_df  = test_df.interpolate(method='linear', limit_direction='both', axis=0)

numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns

scaler = MinMaxScaler(feature_range=(0, 1))

train_df[numerical_columns] = scaler.fit_transform(
    train_df[numerical_columns]
)

val_df[numerical_columns] = scaler.transform(
    val_df[numerical_columns]
)

test_df[numerical_columns] = scaler.transform(
    test_df[numerical_columns]
)

Y_train = train_df.iloc[:, 7:]

X1_train_raw = train_df.iloc[:, [1]]
X2_train_raw = train_df.iloc[:, [2]]
X3_train_raw = train_df.iloc[:, [3]]
X4_train_raw = train_df.iloc[:, [4]]
X5_train_raw = train_df.iloc[:, [5]]
X6_train_raw = train_df.iloc[:, [6]]

time_train_raw = train_df.iloc[:, [0]]

def create_sequences(X1, X2, X3, X4, X5, X6,
                     Y, time, time_steps, horizon):

    X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq = [], [], [], [], [], []
    Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq = [], [], [], [], [], []
    Y_seq = []
    time_seq = []

    for i in range(len(Y) - time_steps - horizon + 1):
        X1_seq.append(X1[i:i+time_steps])
        X2_seq.append(X2[i:i+time_steps])
        X3_seq.append(X3[i:i+time_steps])
        X4_seq.append(X4[i:i+time_steps])
        X5_seq.append(X5[i:i+time_steps])
        X6_seq.append(X6[i:i+time_steps])

        Y1_past_seq.append(Y.iloc[i:i+time_steps, 0])
        Y2_past_seq.append(Y.iloc[i:i+time_steps, 1])
        Y3_past_seq.append(Y.iloc[i:i+time_steps, 2])
        Y4_past_seq.append(Y.iloc[i:i+time_steps, 3])
        Y5_past_seq.append(Y.iloc[i:i+time_steps, 4])
        Y6_past_seq.append(Y.iloc[i:i+time_steps, 5])

        Y_seq.append(Y.iloc[i+time_steps:i+time_steps+horizon, :])

        time_seq.append(time[i:i+time_steps])

    X1_seq = np.array(X1_seq)
    X2_seq = np.array(X2_seq)
    X3_seq = np.array(X3_seq)
    X4_seq = np.array(X4_seq)
    X5_seq = np.array(X5_seq)
    X6_seq = np.array(X6_seq)

    Y1_past_seq = np.array(Y1_past_seq)
    Y2_past_seq = np.array(Y2_past_seq)
    Y3_past_seq = np.array(Y3_past_seq)
    Y4_past_seq = np.array(Y4_past_seq)
    Y5_past_seq = np.array(Y5_past_seq)
    Y6_past_seq = np.array(Y6_past_seq)

    Y_seq = np.array(Y_seq)
    time_seq = np.array(time_seq)

    return (X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq,
            Y1_past_seq, Y2_past_seq, Y3_past_seq, Y4_past_seq, Y5_past_seq, Y6_past_seq,
            Y_seq, time_seq)

(
    X1_train, X2_train, X3_train,
    X4_train, X5_train, X6_train,
    modality_1_train, modality_2_train,
    modality_3_train, modality_4_train,
    modality_5_train, modality_6_train,
    y_train, train_time
) = create_sequences(
    X1_train_raw,
    X2_train_raw,
    X3_train_raw,
    X4_train_raw,
    X5_train_raw,
    X6_train_raw,
    Y_train,
    time_train_raw,
    time_steps,
    horizon
)

Y_val = val_df.iloc[:, 7:]

X1_val_raw = val_df.iloc[:, [1]]
X2_val_raw = val_df.iloc[:, [2]]
X3_val_raw = val_df.iloc[:, [3]]
X4_val_raw = val_df.iloc[:, [4]]
X5_val_raw = val_df.iloc[:, [5]]
X6_val_raw = val_df.iloc[:, [6]]

time_val_raw = val_df.iloc[:, [0]]

(
    X1_val, X2_val, X3_val,
    X4_val, X5_val, X6_val,
    modality_1_val, modality_2_val,
    modality_3_val, modality_4_val,
    modality_5_val, modality_6_val,
    y_val, val_time
) = create_sequences(
    X1_val_raw,
    X2_val_raw,
    X3_val_raw,
    X4_val_raw,
    X5_val_raw,
    X6_val_raw,
    Y_val,
    time_val_raw,
    time_steps,
    horizon
)

Y_test = test_df.iloc[:, 7:]

X1_test_raw = test_df.iloc[:, [1]]
X2_test_raw = test_df.iloc[:, [2]]
X3_test_raw = test_df.iloc[:, [3]]
X4_test_raw = test_df.iloc[:, [4]]
X5_test_raw = test_df.iloc[:, [5]]
X6_test_raw = test_df.iloc[:, [6]]

time_test_raw = test_df.iloc[:, [0]]

(
    X1_test, X2_test, X3_test,
    X4_test, X5_test, X6_test,
    modality_1_test, modality_2_test,
    modality_3_test, modality_4_test,
    modality_5_test, modality_6_test,
    y_test, test_time
) = create_sequences(
    X1_test_raw,
    X2_test_raw,
    X3_test_raw,
    X4_test_raw,
    X5_test_raw,
    X6_test_raw,
    Y_test,
    time_test_raw,
    time_steps,
    horizon
)

# Model parameters
TIME_STEPS = 30 
HORIZON = 7  
N_FEATURES = 12
INPUTS = tf.concat([X1_train,X2_train,X3_train,X4_train,X5_train,X6_train, modality_1_train,modality_2_train,modality_3_train,modality_4_train,modality_5_train, modality_6_train], axis=-1)
N_CNN_FILTERS = 64
CNN_FILTER_SIZE = 3
LSTM_UNITS = 64
ATTENTION_HEADS = 4

# 1DCNN
def cnn1d(input_shape, horizon):
    inputs = Input(shape=input_shape)
    x = Conv1D(
    filters=64,
    padding="causal",
    kernel_size=3,
    activation='relu'
    )(inputs)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(horizon*6)(x)
    outputs = Reshape((horizon,6))(x)
    model = Model(inputs, outputs)
    return model
# Complete model
def build_model(time_steps, n_features, horizon, inputs):
    inputs = Input(shape=(time_steps, n_features)) 
    lstm_outputs = cnn1d((time_steps, n_features),horizon)(inputs)
    model = Model(inputs=inputs, outputs=lstm_outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
def ensemble_predict(models, x): 
    predictions = np.array([model.predict(x) for model in models])  
    avg_predictions = np.mean(predictions, axis=0)  
    return avg_predictions

n_runs = 10
durations = []
durations_predictions = []
models = []
predictionstosave = []

for i in range(n_runs):
    model = build_model(TIME_STEPS, N_FEATURES, HORIZON, INPUTS)  
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    start_time = pytime.time()  
    history = model.fit(INPUTS, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    end_time = pytime.time() 
    duration = end_time - start_time

    durations.append(duration)
    models.append(model)

    best_epoch = np.argmin(history.history['val_loss']) + 1

    test_loss = model.evaluate(tf.concat([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, modality_1_test, modality_2_test, modality_3_test, modality_4_test, modality_5_test, modality_6_test], axis=-1), y_test)
    start_time2 = pytime.time()
    prediction = model.predict(tf.concat([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, modality_1_test, modality_2_test, modality_3_test, modality_4_test, modality_5_test, modality_6_test], axis=-1))
    end_time2  = pytime.time()
    duration2 = end_time2 - start_time2

    durations_predictions.append(duration2)
    model.save(f'Results/Saved-from-run-models/1dcnn-mul-{i}.h5')
    predictionstosave.append(prediction)

np.save(f'Results/Saved-from-run-models/full-predictions-1dcnn-mul.npy', predictionstosave)

scaler_modified = MinMaxScaler(feature_range=(0, 1))
scaler_modified.min_ = scaler.min_[6:]
scaler_modified.scale_ = scaler.scale_[6:]
scaler_modified.data_min_ = scaler.data_min_[6:]
scaler_modified.data_max_ = scaler.data_max_[6:]
scaler_modified.data_range_ = scaler.data_range_[6:]
scaler_modified.feature_range = scaler.feature_range

predictions = ensemble_predict(models, tf.concat([X1_test, X2_test,X3_test,X4_test,X5_test,X6_test,modality_1_test,modality_2_test,modality_3_test,modality_4_test,modality_5_test, modality_6_test], axis=-1))
np.save(f'Results/predictions-1dcnn-mul.npy', predictions)

samples, horizon, stations = y_test.shape
y_test2 = y_test.reshape(-1, stations)
y_test_unscaled = scaler_modified.inverse_transform(y_test2)

rmse_per_run  = []
mse_per_run  = []
mae_per_run  = []
mae_high_per_run = []
mape_per_run = []
nse_per_run  = []
kge_per_run  = []

for preds in predictionstosave:
    samples, horizon, stations = preds.shape
    preds2 = preds.reshape(-1, stations)
    preds_unscaled = scaler_modified.inverse_transform(preds2)

    df_actual = pd.DataFrame(y_test_unscaled.flatten(), columns=['Flow'])
    df_pred = pd.DataFrame(preds_unscaled.flatten(), columns=['Flow'])   

    threshold_high_actual = df_actual['Flow'].quantile(0.95)
    threshold_high_pred = df_actual['Flow'].quantile(0.95)

    extreme_highs_actual = df_actual[df_actual['Flow'] >= threshold_high_actual]
    extreme_highs_pred   = df_pred[df_actual['Flow'] >= threshold_high_pred]

    rmse = root_mean_squared_error(y_test_unscaled.flatten(), preds_unscaled.flatten())    
    mse = mean_squared_error(y_test_unscaled.flatten(), preds_unscaled.flatten())
    mae = mean_absolute_error(y_test_unscaled.flatten(), preds_unscaled.flatten())
    mae_high = mean_absolute_error(extreme_highs_actual, extreme_highs_pred)
    mape = mean_absolute_percentage_error(y_test_unscaled.flatten(), preds_unscaled.flatten())
    nse_value = evaluator(nse, np.array(preds_unscaled.flatten()), np.array(y_test_unscaled.flatten()))
    kge_value = evaluator(kge, np.array(preds_unscaled.flatten()), np.array(y_test_unscaled.flatten()))
    
    rmse_per_run.append(round(rmse, 2))
    mse_per_run.append(round(mse, 2))
    mae_per_run.append(round(mae, 2))
    mae_high_per_run.append(round(mae_high, 2))
    mape_per_run.append(round(mape, 2))
    nse_per_run.append(round(nse_value[0], 2))
    kge_per_run.append(round(kge_value[0,0], 2))

durations_and_metrics_df = pd.DataFrame({
  'Run': [f'{i+1}' for i in range(n_runs)],
    'duration': durations,
    'prediction-duration': durations_predictions,
    'RMSE': rmse_per_run,
    'MSE':  mse_per_run,
    'MAE':  mae_per_run,
    'MAE_HIGH':  mae_high_per_run,
    'MAPE': mape_per_run,
    'NSE':  nse_per_run,
    'KGE':  kge_per_run,            
})

durations_and_metrics_df.to_csv('Results/Saved-from-run-models/durations-and-metrics-1dcnn-mul.csv')