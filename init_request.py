import requests

url = "http://127.0.0.1:5000/init"

payload={'parameters': '''{
    "geography": {
        "latitude": 56.11,
        "longitude": 47.48,
        "timezone": "Europe/Moscow" },
    "pv_array": {
        "pv_tilt_angle": 40,
        "pv_azimuth_angle": 180,
        "pv_modules_number": 1,
        "pv_module_params": { "stc_power": 298.5, "area": 1.5, "kt": -0.38} },

    "training": {
    "predictors": ["aoi","T","U","Сl"],
    "predictand": "I"},

    "files_destinations": {
        "trained_model": ".\\\\forecasting\\\\trained_model.mdl",
        "X_scaler_params": ".\\\\forecasting\\\\X_scaler_params.sca",
        "y_scaler_params": ".\\\\forecasting\\\\y_scaler_params.sca",
        "training_dataset": ".\\\\forecasting\\\\training_dataset_nvchb.csv",
        "new_X_scaler_filename": ".\\\\forecasting\\\\new_X_scaler_params.sca",
        "new_y_scaler_filename": ".\\\\forecasting\\\\new_y_scaler_params.sca",
        "new_trained_model_filename": ".\\\\forecasting\\\\new_trained_model.mdl",
        "updating_dataset": ".\\\\forecasting\\\\updating_dataset_nvchb.csv" }
}'''
}

files=[
  ('files[]',('trained_model.mdl', open('файлы для инициализации/trained_model.mdl', 'rb'), 'application/octet-stream')),
  ('files[]',('training_dataset_nvchb.csv',open('файлы для инициализации/training_dataset_nvchb.csv','rb'),'text/csv')),
  ('files[]',('updating_dataset_nvchb.csv',open('файлы для инициализации/updating_dataset_nvchb.csv','rb'),'text/csv')),
  ('files[]',('X_scaler_params.sca',open('файлы для инициализации/X_scaler_params.sca','rb'),'application/octet-stream')),
  ('files[]',('y_scaler_params.sca',open('файлы для инициализации/y_scaler_params.sca','rb'),'application/octet-stream'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
