import time as pytime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from hydroeval import evaluator, nse, kge
import argparse
import joblib as joblib


parser = argparse.ArgumentParser(description="Run river flow prediction model.")
parser.add_argument('--model', type=str, default="svr", help='Model. Choose from ["rf", "svr"]')
parser.add_argument('--timesteps', type=int, default=30, help='Number of time steps for the input sequences')
parser.add_argument('--horizon', type=int, default=7, help='Prediction horizon')
parser.add_argument('--n_runs', type=int, default=10, help='Model runs')
args = parser.parse_args()

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

split_index = int(len(weather_data) * 0.8)

train_df = weather_data.iloc[:split_index]
test_df  = weather_data.iloc[split_index:]
num_of_targets = len(train_df.iloc[:,7:13].columns)
print(num_of_targets)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(train_df.iloc[:,1:7])
X_test  = x_scaler.transform(test_df.iloc[:,1:7])

X_train = train_df.iloc[:,1:7]
X_test = test_df.iloc[:,1:7]

X_train = X_train.interpolate(method='linear', limit_direction='both', axis=0)
X_test = X_test.interpolate(method='linear', limit_direction='both', axis=0)

Y_train = train_df.iloc[:,7:13]
Y_test = test_df.iloc[:,7:13]

Y_train = Y_train.interpolate(method='linear', limit_direction='both', axis=0)
Y_test = Y_test.interpolate(method='linear', limit_direction='both', axis=0)

Y_train = y_scaler.fit_transform(train_df.iloc[:,7:13])
Y_test  = y_scaler.transform(test_df.iloc[:,7:13])

time_train = train_df.iloc[:,0].values
time_test = test_df.iloc[:,0].values

def create_sequences(X, Y, time, time_steps, horizon):
    
    X_seq = []
    Y_past_seq = []
    Y_seq = []
    time_seq = []

    for i in range(len(X) - time_steps - horizon + 1):

        X_seq.append(X[i:i+time_steps])

        Y_past_seq.append(Y[i:i+time_steps])

        Y_seq.append(Y[i+time_steps:i+time_steps+horizon])

        time_seq.append(time[i:i+time_steps])

    return (
        np.array(X_seq),
        np.array(Y_past_seq),
        np.array(Y_seq),
        np.array(time_seq)
    )

X_train_seq, Y_past_train_seq, Y_train_seq, train_time_seq = create_sequences(
    X_train, Y_train, time_train, time_steps, horizon
)

X_test_seq, Y_past_test_seq, Y_test_seq, test_time_seq = create_sequences(
    X_test, Y_test, time_test, time_steps, horizon
)

y_train_sets = [Y_train_seq[:,:,i] for i in range(num_of_targets)]
y_test_sets  = [Y_test_seq[:,:,i]  for i in range(num_of_targets)]

y_past_train_sets = Y_past_train_seq

y_past_test_sets = Y_past_test_seq

# Model parameters
TIME_STEPS = 30 
HORIZON = 7  
N_FEATURES = 12
INPUTS = np.concatenate([X_train_seq, y_past_train_sets], axis=-1)
N_MODALS = 12

class SklearnWrapper:
    def __init__(self, model_type='svr', seed=42):
        self.model_type = model_type
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                    n_estimators=50,       
                    max_depth=5,
                    min_samples_leaf=5,
                    min_samples_split=5,
                    max_features=0.15,      
                    max_samples=0.8,
                    n_jobs=-1,             
                    random_state=seed      
                )
        elif model_type == 'svr':
            self.model = MultiOutputRegressor(
                SVR(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    epsilon=0.1,
                    max_iter=5000
                )
            )
    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        self.model.fit(X_flat, y_flat)
        return self 

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        preds = self.model.predict(X_flat)
        return preds.reshape(-1, 7, 6) 

    def save(self, path):
        joblib.dump(self.model, path)
        
def ensemble_predict(models, x): 
    predictions = np.array([model.predict(x) for model in models])  
    avg_predictions = np.mean(predictions, axis=0)  
    return avg_predictions

n_runs = args.n_runs
durations = []
durations_predictions = []
models = []
predictionstosave = []

for i in range(n_runs):
    
    current_seed = i + 42 
    np.random.seed(current_seed)

    if args.model in ['svr','rf']:
        train_len = int(len(INPUTS) * 0.80)
        max_start = len(INPUTS) - train_len
        start_idx = np.random.randint(0, max_start)
        
        X_train_sub = INPUTS[start_idx : start_idx + train_len]
        Y_train_sub = y_train[start_idx : start_idx + train_len]

        model = SklearnWrapper(model_type=args.model, seed=current_seed)
        
        start_time = pytime.time()
        model.fit(X_train_sub, Y_train_sub) 
        duration = pytime.time() - start_time
        start_time2 = pytime.time()
        X_test_combined = np.concatenate([X_test_seq, y_past_test_sets], axis=-1)
        prediction = model.predict(X_test_combined)
        duration2 = pytime.time() - start_time2
        predictionstosave.append(prediction)

np.save(f'Results/Saved-from-run-models/full-predictions-{args.model}.npy', predictionstosave)

predictions = ensemble_predict(models, np.concatenate([X_test_seq, y_past_test_sets], axis=-1))

np.save(f'Results/Saved-from-run-models/predictions-{args.model}.npy', predictions)

samples, horizon, stations = y_test.shape
y_test2 = y_test.reshape(-1, stations)
y_test_unscaled = y_test2
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
    preds_unscaled = preds2

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

durations_and_metrics_df.to_csv(f'Results/Saved-from-run-models/durations-and-metrics-{args.model}.csv')


