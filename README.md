# StreamflowForecastingNN-DynFus

This is the repository with the LSTM Encoder-Decoder **dynamic feature fusion models** for high-flow forecasting using ERA5-Land climatic data and river flow measurement data.

## Project structure
```
.
├── Data/
│   └── Final-data/
│
├── Models/
│   ├── XAI/
│   ├── encdec_dyn1_fus_mul.py
│   ├── encdec_dyn1_fus_un.py
│   ├── encdec_dyn_fus_mul.py
│   ├── encdec_dyn_fus_un.py
│   ├── bas_1dcnn.py
│   ├── bas_rf_and_svr.py
│   ├── requirements.txt
│   └── .Dockerfile
│
├── Results/
│   ├── Saved-from-run-models/
│   ├── mul-vs-un-results-analysis.ipynb
│   ├── multivariate-results-analysis.ipynb
│   ├── univariate-results-analysis.ipynb
│   └── EDA-and-model-weights-analysis.ipynb
│
├── README.md
└── LICENSE
```

## Data

The `Data/` folder contains:

- ERA5-Land data
- River flow measurements from the DEFRA/NRFA APIs

## Model development

The models use the data in `Data/Final-data/` as input. All model scripts are located in the `Models/` folder.

## Results

The `Results/` folder contains the Jupyter notebooks that analyse the model predictions. These notebooks use the saved outputs in `Results/Saved-from-run-models/` to generate visualisations, graphs and tables. The results reported in the paper can be reproduced by running the notebook cells.

## Requirements

- TensorFlow 2.11.0 (GPU), CUDA 11.2

The remaining Python dependencies are listed in `Models/requirements.txt`.

## Running models

Docker is optional. To run a model without Docker, skip steps 3–4 and install the dependencies instead:
```bash
pip install -r requirements.txt
```

1. Clone the repository.
2. Navigate to the `Models/` folder:
   ```bash
   cd Models
   ```
4. Build the Docker image:
   ```bash
   docker build -f .Dockerfile -t <IMAGE_NAME> .
   ```
5. Run the container interactively:
   ```bash
   docker run --gpus device=0 \
     -v "$(pwd)/../Data":/app/../Data \
     -v "$(pwd)/../Results":/app/../Results \
     -it <IMAGE_NAME> bash
   ```
6. Run a model, for example:
   ```bash
   python3 encdec_dyn1_fus_mul.py
   ```

The model outputs are saved in `Results/Saved-from-run-models/`.

## Explainability

The `Models/XAI/` folder includes the scripts for **KernelSHAP** explainability. To run one, from the `Models/` folder:

```bash
python3 XAI/kernelshap_val_<type_of_plot>.py
```

where `<type_of_plot>` is replaced by the plot type in the script name.
