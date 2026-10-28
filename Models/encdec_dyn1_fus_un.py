import time as pytime
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Multiply, MultiHeadAttention, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from hydroeval import evaluator, nse, kge


folder_with_data = 'Data/Final-data/'
river_flow = pd.read_csv(folder_with_data + '/river-flow.csv')
river_flow['dateTime'] = pd.to_datetime(river_flow['dateTime'], errors='raise')
river_flow['dateTime'] = river_flow['dateTime'].dt.date
river_flow = river_flow.groupby('dateTime').mean().reset_index()

climate_variables = pd.read_csv(folder_with_data + '/atmospheric-variables.csv')
climate_variables['dateTime'] = pd.to_datetime(climate_variables['dateTime'], errors='raise')
climate_variables['dateTime'] = climate_variables['dateTime'].dt.date

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

Y1_train_raw = train_df.iloc[:, [7]]
Y2_train_raw = train_df.iloc[:, [8]]
Y3_train_raw = train_df.iloc[:, [9]]
Y4_train_raw = train_df.iloc[:, [10]]
Y5_train_raw = train_df.iloc[:, [11]]
Y6_train_raw = train_df.iloc[:, [12]]

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
    
    Y_past_seq = []
    
    Y_seq = []
    
    time_seq = []

    for i in range(len(Y) - time_steps - horizon + 1):
        X1_seq.append(X1[i:i+time_steps])
        X2_seq.append(X2[i:i+time_steps])
        X3_seq.append(X3[i:i+time_steps])
        X4_seq.append(X4[i:i+time_steps])
        X5_seq.append(X5[i:i+time_steps])
        X6_seq.append(X6[i:i+time_steps])
        
        Y_past_seq.append(Y.iloc[i:i+time_steps]) 

        Y_seq.append(Y.iloc[i+time_steps:i+time_steps+horizon])  

        time_seq.append(time[i:i+time_steps])

    X1_seq = np.array(X1_seq)
    X2_seq = np.array(X2_seq)
    X3_seq = np.array(X3_seq)
    X4_seq = np.array(X4_seq)
    X5_seq = np.array(X5_seq)
    X6_seq = np.array(X6_seq)
    
    Y_past_seq = np.array(Y_past_seq)
    
    Y_seq = np.array(Y_seq)
    
    time_seq = np.array(time_seq)

    return (X1_seq, X2_seq, X3_seq, X4_seq, X5_seq, X6_seq, 
            Y_past_seq,
            Y_seq, time_seq)

Y1_val_raw = val_df.iloc[:, [7]]
Y2_val_raw = val_df.iloc[:, [8]]
Y3_val_raw = val_df.iloc[:, [9]]
Y4_val_raw = val_df.iloc[:, [10]]
Y5_val_raw = val_df.iloc[:, [11]]
Y6_val_raw = val_df.iloc[:, [12]]

Y1_test_raw = test_df.iloc[:, [7]]
Y2_test_raw = test_df.iloc[:, [8]]
Y3_test_raw = test_df.iloc[:, [9]]
Y4_test_raw = test_df.iloc[:, [10]]
Y5_test_raw = test_df.iloc[:, [11]]
Y6_test_raw = test_df.iloc[:, [12]]

X1_test_raw = test_df.iloc[:, [1]]
X2_test_raw = test_df.iloc[:, [2]]
X3_test_raw = test_df.iloc[:, [3]]
X4_test_raw = test_df.iloc[:, [4]]
X5_test_raw = test_df.iloc[:, [5]]
X6_test_raw = test_df.iloc[:, [6]]

time_test_raw = test_df.iloc[:, [0]]

X1_val_raw = val_df.iloc[:, [1]]
X2_val_raw = val_df.iloc[:, [2]]
X3_val_raw = val_df.iloc[:, [3]]
X4_val_raw = val_df.iloc[:, [4]]
X5_val_raw = val_df.iloc[:, [5]]
X6_val_raw = val_df.iloc[:, [6]]

time_val_raw = val_df.iloc[:, [0]]

(   X1_train, X2_train, X3_train,
    X4_train, X5_train, X6_train,
    y1_past_train, y1_train,
    train_time
) = create_sequences(
    X1_train_raw,
    X2_train_raw,
    X3_train_raw,
    X4_train_raw,
    X5_train_raw,
    X6_train_raw,
    Y1_train_raw,
    time_train_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, \
y2_past_train, y2_train, _ = create_sequences(
    X1_train_raw,
    X2_train_raw,
    X3_train_raw,
    X4_train_raw,
    X5_train_raw,
    X6_train_raw,
    Y2_train_raw,
    time_train_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, \
y3_past_train, y3_train, _ = create_sequences(
    X1_train_raw,
    X2_train_raw,
    X3_train_raw,
    X4_train_raw,
    X5_train_raw,
    X6_train_raw,
    Y3_train_raw,
    time_train_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, \
y4_past_train, y4_train, _ = create_sequences(
    X1_train_raw,
    X2_train_raw,
    X3_train_raw,
    X4_train_raw,
    X5_train_raw,
    X6_train_raw,
    Y4_train_raw,
    time_train_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, \
y5_past_train, y5_train, _ = create_sequences(
    X1_train_raw,
    X2_train_raw,
    X3_train_raw,
    X4_train_raw,
    X5_train_raw,
    X6_train_raw,
    Y5_train_raw,
    time_train_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, \
y6_past_train, y6_train, _ = create_sequences(
    X1_train_raw,
    X2_train_raw,
    X3_train_raw,
    X4_train_raw,
    X5_train_raw,
    X6_train_raw,
    Y6_train_raw,
    time_train_raw,
    time_steps,
    horizon
)

modality_1_train = y1_past_train
modality_2_train = y2_past_train
modality_3_train = y3_past_train
modality_4_train = y4_past_train
modality_5_train = y5_past_train
modality_6_train = y6_past_train

(
    X1_val, X2_val, X3_val,
    X4_val, X5_val, X6_val,
    y1_past_val, y1_val,
    val_time
) = create_sequences(
    X1_val_raw,
    X2_val_raw,
    X3_val_raw,
    X4_val_raw,
    X5_val_raw,
    X6_val_raw,
    Y1_val_raw,
    time_val_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y2_past_val, y2_val, _ = create_sequences(
    X1_val_raw,
    X2_val_raw,
    X3_val_raw,
    X4_val_raw,
    X5_val_raw,
    X6_val_raw,
    Y2_val_raw,
    time_val_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y3_past_val, y3_val, _ = create_sequences(
    X1_val_raw,
    X2_val_raw,
    X3_val_raw,
    X4_val_raw,
    X5_val_raw,
    X6_val_raw,
    Y3_val_raw,
    time_val_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y4_past_val, y4_val, _ = create_sequences(
    X1_val_raw,
    X2_val_raw,
    X3_val_raw,
    X4_val_raw,
    X5_val_raw,
    X6_val_raw,
    Y4_val_raw,
    time_val_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y5_past_val, y5_val, _ = create_sequences(
    X1_val_raw,
    X2_val_raw,
    X3_val_raw,
    X4_val_raw,
    X5_val_raw,
    X6_val_raw,
    Y5_val_raw,
    time_val_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y6_past_val, y6_val, _ = create_sequences(
    X1_val_raw,
    X2_val_raw,
    X3_val_raw,
    X4_val_raw,
    X5_val_raw,
    X6_val_raw,
    Y6_val_raw,
    time_val_raw,
    time_steps,
    horizon
)

modality_1_val = y1_past_val
modality_2_val = y2_past_val
modality_3_val = y3_past_val
modality_4_val = y4_past_val
modality_5_val = y5_past_val
modality_6_val = y6_past_val

(
    X1_test, X2_test, X3_test,
    X4_test, X5_test, X6_test,
    y1_past_test, y1_test,
    test_time
) = create_sequences(
    X1_test_raw,
    X2_test_raw,
    X3_test_raw,
    X4_test_raw,
    X5_test_raw,
    X6_test_raw,
    Y1_test_raw,
    time_test_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y2_past_test, y2_test, _ = create_sequences(
    X1_test_raw,
    X2_test_raw,
    X3_test_raw,
    X4_test_raw,
    X5_test_raw,
    X6_test_raw,
    Y2_test_raw,
    time_test_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y3_past_test, y3_test, _ = create_sequences(
    X1_test_raw,
    X2_test_raw,
    X3_test_raw,
    X4_test_raw,
    X5_test_raw,
    X6_test_raw,
    Y3_test_raw,
    time_test_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y4_past_test, y4_test, _ = create_sequences(
    X1_test_raw,
    X2_test_raw,
    X3_test_raw,
    X4_test_raw,
    X5_test_raw,
    X6_test_raw,
    Y4_test_raw,
    time_test_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y5_past_test, y5_test, _ = create_sequences(
    X1_test_raw,
    X2_test_raw,
    X3_test_raw,
    X4_test_raw,
    X5_test_raw,
    X6_test_raw,
    Y5_test_raw,
    time_test_raw,
    time_steps,
    horizon
)
_, _, _, _, _, _, y6_past_test, y6_test, _ = create_sequences(
    X1_test_raw,
    X2_test_raw,
    X3_test_raw,
    X4_test_raw,
    X5_test_raw,
    X6_test_raw,
    Y6_test_raw,
    time_test_raw,
    time_steps,
    horizon
)

modality_1_test = y1_past_test
modality_2_test = y2_past_test
modality_3_test = y3_past_test
modality_4_test = y4_past_test
modality_5_test = y5_past_test
modality_6_test = y6_past_test

# Model parameters
TIME_STEPS = 30 
HORIZON = 7  
N_FEATURES = 1
N_MODALS = 7
N_CNN_FILTERS = 64
CNN_FILTER_SIZE = 3
LSTM_UNITS = 64
ATTENTION_HEADS = 4

def dynamic_fusion_cell(inputs,
                        num_modalities,
                        stations=1):

    concatenated = Concatenate(name='concat')(inputs)
    
    multiplied = Multiply(name='hadamard_mul')(inputs)
    multiplied = Dense(units=num_modalities, activation='relu')(multiplied)
    
    gate_weights = Dense(
        num_modalities,
        activation='softmax'
    )(concatenated)
    weighted = Multiply(name='weighted')(
        [concatenated, gate_weights]
    )

    added = tf.reduce_sum(concatenated, axis=2)
    added = tf.expand_dims(added, axis=-1)
    added = Dense(
        num_modalities,
        activation='relu',
        name='addition'
    )(added)

    all_fus_rep = tf.stack(
        [concatenated, weighted, added, multiplied],
        axis=0
    )
    all_fus_rep = tf.transpose(
        all_fus_rep,
        perm=[1, 2, 3, 0]
    )

    fusion_weights = Dense(4,
        activation='softmax',
        name='fusion'
    )(all_fus_rep)

    fin_weighted = Multiply()(
        [all_fus_rep, fusion_weights]
    )
    final = tf.reduce_sum(fin_weighted, axis=3)
    final = Dense(stations, name='forecast')(final)

    return final, gate_weights, fusion_weights
# LSTM Encoder-Decoder
def lstm_encoder_decoder(input_shape, horizon):
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation='relu')(inputs)
    x = RepeatVector(horizon)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)
    model = Model(inputs, x)
    return model
# Multi-Head Attention
def attention_block(lstm_output):
    output = MultiHeadAttention(num_heads=ATTENTION_HEADS, key_dim=LSTM_UNITS // ATTENTION_HEADS)(lstm_output, lstm_output, lstm_output)
    return output

# class TestLossHistory(tf.keras.callbacks.Callback):
#     def __init__(self, test_data):
#         super().__init__()
#         self.test_data = test_data
#         self.test_losses = []

#     def on_epoch_end(self, epoch, logs=None):
#         test_loss = self.model.evaluate(
#             self.test_data[0],
#             self.test_data[1],
#             verbose=0
#         )
#         self.test_losses.append(test_loss)
#         print(f"Epoch {epoch+1}: test_loss = {test_loss:.6f}")

# Complete model
def build_model(time_steps, n_features, n_modals, horizon, inputs):
    inputs = [Input(shape=(time_steps, n_features)) for _ in range(n_modals)]
    lstm_outputs = [lstm_encoder_decoder((time_steps,n_features),horizon)(mod_input) for mod_input in inputs]
    attention_outputs = [attention_block(lstm_out) for lstm_out in lstm_outputs]
    fusion_output = dynamic_fusion_cell(attention_outputs, n_modals) 
    model = Model(inputs=inputs, outputs=fusion_output)
    model.compile(optimizer='adam' , loss='mean_squared_error')
    return model
def ensemble_predict(predictions): 
    predictions = np.array(predictions)
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions

n_runs = 10                  
n_stations = 6              
models_per_run = []          
predictions_per_run = []     
durations_per_run = []   
durations_for_preds_per_run = []   
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
y_train_sets = [y1_train, y2_train, y3_train, y4_train, y5_train, y6_train]
y_test_sets = [y1_test, y2_test, y3_test, y4_test, y5_test, y6_test]
y_past_train_sets = [y1_past_train, y2_past_train, y3_past_train, y4_past_train, y5_past_train, y6_past_train]
y_past_test_sets = [y1_past_test, y2_past_test, y3_past_test, y4_past_test, y5_past_test, y6_past_test]
station_ids = ['54001', '54005', '54032','54057', '54095', '55002']

for run in range(n_runs):
    run_models = []             
    run_predictions = []
    # run_histories = []
    # run_test_histories = []

    for i, (y_train, y_past_train) in enumerate(zip(y_train_sets, y_past_train_sets)):
        INPUTS = [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, y_past_train]
        model = build_model(TIME_STEPS, N_FEATURES, N_MODALS, HORIZON, INPUTS)
        model.summary()
        run_models.append(model)
    
    start_time = pytime.time() 

    for model, y_train, y_past_train in zip(run_models, y_train_sets, y_past_train_sets):
        # test_callback = TestLossHistory(test_data=([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, y_past_test], y_test))
        history = model.fit([X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, y_past_train], 
                            y_train, epochs=50, batch_size=32, validation_split=0.2, 
                            callbacks=[early_stopping
                                    #    ,test_callback
                                       ])
        # run_histories.append(history)
        # run_test_histories.append(test_callback)
    
    end_time = pytime.time()
    duration = end_time - start_time
    durations_per_run.append(duration)

    for model, y_test, y_past_test in zip(run_models, y_test_sets, y_past_test_sets):
        test_loss = model.evaluate([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, y_past_test], y_test)

    start_time2 = pytime.time() 

    for model, y_past_test in zip(run_models, y_past_test_sets):
        predictions = model.predict([X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, y_past_test])
        run_predictions.append(predictions)

    end_time2 = pytime.time()
    duration2 = end_time2 - start_time2
    durations_for_preds_per_run.append(duration2)

    # for i, history in enumerate(run_histories):

    #     train_losses = history.history.get('loss', [])
    #     val_losses = history.history.get('val_loss', [])
    #     test_losses = run_test_histories[i].test_losses  

    #     best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else len(train_losses)
    #     best_test_loss = test_losses[best_epoch - 1] if test_losses else None

    #     fig, ax = plt.subplots(figsize=(15, 10), dpi=500)
    #     epochs_axis = np.arange(1, len(train_losses) + 1)
    #     ax.plot(epochs_axis, train_losses, label="Train loss", linewidth=3, color="#1f77b4")
    #     if val_losses:
    #         ax.plot(epochs_axis, val_losses, label="Validation loss", linewidth=3, color="#ff7f0e")
    #     if test_losses:
    #         ax.plot(
    #             np.arange(1, len(test_losses) + 1),
    #             test_losses,
    #             label="Test loss",
    #             linewidth=3,
    #             color="#2ca02c")
    #     ax.axvline(best_epoch, color="red", linestyle=":", linewidth=2, alpha=0.8)
    #     ax.annotate(
    #             f"Epoch {best_epoch}\nTest loss={best_test_loss:.4f}",
    #             xy=(best_epoch, best_test_loss),
    #             xytext=(best_epoch + 0.005, best_test_loss + 0.0001),
    #             bbox=dict(
    #                 boxstyle="round,pad=0.4",
    #                 fc="white",
    #                 ec="#2ca02c",
    #                 alpha=0.9),
    #             arrowprops=dict(arrowstyle="->", color="#2ca02c"),
    #             fontsize=22,
    #             color="#2ca02c")
    #     if val_losses:
    #         ax.scatter([best_epoch], [val_losses[best_epoch - 1]], color="#d62728", s=160, zorder=5, label="Early stop. val. epoch")
    #     ax.set_ylabel("Loss", fontsize=30, fontweight="bold")
    #     ax.set_xlabel("Epoch", fontsize=30, fontweight="bold")
    #     ax.set_title(f"Station {station_ids[i]}", fontsize=30, fontweight="bold")
    #     ax.tick_params(axis="x", labelsize=20, rotation=0)
    #     ax.tick_params(axis="y", labelsize=20)
    #     ax.legend(fontsize=20, ncol=1, loc="upper right")
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(2)
    #     ax.grid(True, linestyle="--", alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(
    #         f"Results/Saved-from-run-models/learning-curve-station-{i}.png"
    #     )
    #     plt.close()

    predictions_per_run.append(run_predictions)
    for i, model in enumerate(run_models):
        model.save(f'Results/Saved-from-run-models/encdec-un-dyn1-fusion-run-{run}-station-{i}.h5')

final_ensemble_predictions = []
for station_idx in range(n_stations):
    station_predictions_across_runs = [predictions_per_run[run][station_idx] for run in range(n_runs)]
    station_ensemble = ensemble_predict(station_predictions_across_runs)
    final_ensemble_predictions.append(station_ensemble)

np.save('Results/Saved-from-run-models/predictions-per-run-encdec-dyn1-fusion.npy', final_ensemble_predictions)
np.save('Results/Saved-from-run-models/full-predictions-per-run-encdec-dyn1-fusion.npy', predictions_per_run)
np.save('Results/Saved-from-run-models/durations-encdec-dyn1-fusion.npy', durations_per_run)
np.save('Results/Saved-from-run-models/durations-for-predictions-encdec-dyn1-fusion.npy', durations_for_preds_per_run)

scaler_modified = MinMaxScaler(feature_range=(0, 1))
scaler_modified.feature_range = scaler.feature_range

ys_test = []
for j, y_test in enumerate([y1_test, y2_test, y3_test, y4_test, y5_test, y6_test]): 
    scaler_modified.min_ =  scaler.min_[6+j]
    scaler_modified.scale_ =  scaler.scale_[6+j]
    scaler_modified.data_min_ =  scaler.data_min_[6+j]
    scaler_modified.data_max_ =  scaler.data_max_[6+j]
    scaler_modified.data_range_ =  scaler.data_range_[6+j]

    y_test = scaler_modified.inverse_transform(y_test)
    ys_test.append(y_test)

predictions_per_run = np.array(predictions_per_run)
predictions_per_run2 =[]

for i in range(predictions_per_run.shape[0]):  
    batch_inverse = [] 
    for j in range(predictions_per_run.shape[1]): 
        data_slice = predictions_per_run[i, j, :, :, 0]
        scaler_modified.min_ = scaler.min_[6+j]
        scaler_modified.scale_ = scaler.scale_[6+j]
        scaler_modified.data_min_ = scaler.data_min_[6+j]
        scaler_modified.data_max_ = scaler.data_max_[6+j]
        scaler_modified.data_range_ = scaler.data_range_[6+j]

        data_inverse_transformed = scaler_modified.inverse_transform(data_slice)
        
        batch_inverse.append(data_inverse_transformed)
    
    predictions_per_run2.append(batch_inverse)

predictions_per_run2 = np.array(predictions_per_run2)

y_test_combined = np.stack(ys_test, axis=1) 
y_test_combined = y_test_combined.transpose(0,2,1)

rmse_per_run  = []
mse_per_run  = []
mae_per_run = []
mae_high_per_run  = []
mape_per_run = []
nse_per_run = []
kge_per_run = []

for run_idx, prediction_per_run in enumerate(predictions_per_run2): 
    preds_unscaled = prediction_per_run.transpose(1,2,0).reshape(-1,6).flatten()
    y_test_unscaled = y_test_combined.reshape(-1,6).flatten()

    df_actual = pd.DataFrame(y_test_unscaled, columns=['Flow'])
    df_pred = pd.DataFrame(preds_unscaled, columns=['Flow'])   

    threshold_high_actual = df_actual['Flow'].quantile(0.95)
    threshold_high_pred = df_actual['Flow'].quantile(0.95)

    extreme_highs_actual = df_actual[df_actual['Flow'] >= threshold_high_actual]
    extreme_highs_pred   = df_pred[df_actual['Flow'] >= threshold_high_actual]

    rmse = mean_squared_error(y_test_unscaled, preds_unscaled, squared=False)
    mse = mean_squared_error(y_test_unscaled, preds_unscaled, squared=True)
    mae = mean_absolute_error(y_test_unscaled, preds_unscaled)
    mae_high = mean_absolute_error(extreme_highs_actual, extreme_highs_pred)
    mape = mean_absolute_percentage_error(y_test_unscaled, preds_unscaled)
    nse_value = evaluator(nse, preds_unscaled, y_test_unscaled)
    kge_value = evaluator(kge, preds_unscaled, y_test_unscaled)

    rmse_per_run.append(round(rmse, 2))
    mse_per_run.append(round(mse, 2))
    mae_per_run.append(round(mae, 2))
    mae_high_per_run.append(round(mae_high, 2))
    mape_per_run.append(round(mape, 2))
    nse_per_run.append(round(nse_value[0], 2))  
    kge_per_run.append(round(kge_value[0, 0], 2))

durations_and_metrics_df = pd.DataFrame({
    'Run': [f'{i+1}' for i in range(len(mae_per_run))],
    'duration': durations_for_preds_per_run, 
    'prediction-duration': durations_for_preds_per_run,
    'RMSE': rmse_per_run,
    'MSE':  mse_per_run,
    'MAE': mae_per_run,
    'MAE_HIGH':  mae_high_per_run,
    'MAPE': mape_per_run,
    'NSE': nse_per_run,
    'KGE': kge_per_run,
})

durations_and_metrics_df.to_csv('Results/Saved-from-run-models/durations-and-metrics-encdec-un-dyn1-fusion.csv')



