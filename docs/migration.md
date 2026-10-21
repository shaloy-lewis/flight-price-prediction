# Migration from the original project

Schema/model version 2 deliberately breaks the old implicit artifact contract.
Retrain from the raw source files; do not rename model.pkl or preprocessor.pkl into
the new artifact directory. The old pickles and original notebook are historical
references, excluded from Docker and never loaded by the new application.

1. Install Python 3.12 and the locked dependencies.
2. Retrieve the dataset, then run the validation and training commands in the README.
3. Use artifacts/v2, or set MODEL_DIR to another **trusted local** run directory.
4. Send the canonical schema to /predict. The price response key
   `flight_price_prediction` remains and is in INR.
5. Old request fields are accepted at /predict/legacy. Dates must be DD-MM-YYYY;
   the adapter validates and ignores travel date and emits a warning. Carrier/code
   and clock/duration inconsistencies now return 422. Calls to the original /predict
   payload need to change their path or migrate their schema.
6. Old raw/encoded importance dictionaries are removed. Optional `?explain=true`
   returns labeled attributions with their correct INR or log(INR) scale. Offline
   permutation importance is stored in the run, not recomputed per request.

Python clients may retain CustomData(**legacy_fields).get_data_as_dataframe(), then
call PredictPipeline.predict() on its canonical output. The old preprocess_data,
separate transformation/training methods and custom pickle helpers are replaced by
the shared canonical schema and CLI; downstream imports must migrate accordingly.

A missing model permits API startup for liveness but returns 503 readiness/predictions.
Streamlit displays setup guidance until a compatible model is configured. Restart the
API/clear the Streamlit resource cache when switching model versions.
