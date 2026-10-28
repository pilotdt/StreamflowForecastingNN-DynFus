import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import seaborn as sns

folder_with_data = 'Data'
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

inputs_combined = np.concatenate(
    [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, 
    modality_1_train.reshape(-1, 30, 1), modality_2_train.reshape(-1, 30, 1),
    modality_3_train.reshape(-1, 30, 1), modality_4_train.reshape(-1, 30, 1),
    modality_5_train.reshape(-1, 30, 1), modality_6_train.reshape(-1, 30, 1)],
    axis=-1
)
inputs_test_list = np.concatenate([
    X1_test, X2_test, X3_test, X4_test, X5_test, X6_test,
    modality_1_test.reshape(-1, 30, 1), modality_2_test.reshape(-1, 30, 1),
    modality_3_test.reshape(-1, 30, 1), modality_4_test.reshape(-1, 30, 1),
    modality_5_test.reshape(-1, 30, 1), modality_6_test.reshape(-1, 30, 1)
], axis=-1)

full_feature_names = [f"{name}_timestep_{t}" for t in range(1, 31) for name in numerical_columns]

for model_name in ["encdec-mul-dyn1-fusion"]:
    
    model = load_model(f'../../Results/Saved-from-run-models/{model_name}.h5')

    def model_predict_wrapper(inputs_flat):
        num_samples = inputs_flat.shape[0]
        x_reshaped = inputs_flat.reshape(num_samples, 30, 12)
        input_list = [x_reshaped[:, :, i] for i in range(12)]
        preds = model.predict(input_list, verbose=0)
        return preds[:, 0, 0]

    X_train_flat = inputs_combined.transpose(0, 1, 2).reshape((inputs_combined.shape[0], -1))
    X_test_flat = inputs_test_list.transpose(0, 1, 2).reshape((inputs_test_list.shape[0], -1))  
    
    test_flows = y_test[:, :, 0].mean(axis=1) 
    
    low_threshold = np.percentile(test_flows, 5)
    high_threshold = np.percentile(test_flows, 95)

    high_flow_indices = np.where(test_flows >= high_threshold)[0]
    normal_flow_indices = np.where((test_flows >= low_threshold) & (test_flows < high_threshold))[0]
    all_flow_indices = np.arange(len(test_flows))

    np.random.seed(42)
    idx_high = np.random.choice(high_flow_indices, size=min(100, len(high_flow_indices)), replace=False)
    idx_normal = np.random.choice(normal_flow_indices, size=min(100, len(normal_flow_indices)), replace=False)
    idx_all = np.random.choice(all_flow_indices, size=min(100, len(all_flow_indices)), replace=False)

    regimes = {"high": idx_high, "normal": idx_normal, "all": idx_all}
    
    background = shap.kmeans(X_train_flat, 200) 
    explainer = shap.KernelExplainer(model_predict_wrapper, background)

    for regime_name, indices in regimes.items():
        if len(indices) == 0:
            continue
            
        test_input = X_test_flat[indices]
        shap_values = explainer.shap_values(test_input)
        
        shap_array = np.array(shap_values).reshape(-1, 30, 12)
        feature_contributions = shap_array.sum(axis=1)  
        
        test_input_reshaped = test_input.reshape(-1, 30, 12)
        feature_values_df = pd.DataFrame(test_input_reshaped.sum(axis=1), columns=numerical_columns)

        feature_df = pd.DataFrame(feature_contributions, columns=numerical_columns)
        melted_feature_df = feature_df.melt(var_name="Feature", value_name="SHAP")
        plt.figure(figsize=(8, 10))
        sns.stripplot(data=melted_feature_df, y="Feature", x="SHAP", alpha=0.6, jitter=0.25, color="cornflowerblue", orient="h")
        sns.pointplot(data=melted_feature_df, y="Feature", x="SHAP", estimator=np.mean, color="red", markers="o", linestyles="", orient="h")
        plt.xlabel("SHAP value", fontsize=25)
        plt.ylabel("Feature", fontsize=25)
        plt.xticks(fontsize=18, rotation=45)
        plt.yticks(fontsize=18)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'Results/Saved-from-run-models/shap-jitter-per-feature-{regime_name}-{model_name}.png', dpi=300)
        plt.close()