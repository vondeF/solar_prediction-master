"""
Сюда нужно включить код работы с модулем прогнозирования
В методе init должна проводиться инициализация модуля прогнозирования
 - загрузка исходных данных для обучения и обучение модели
 - загрузка параметров обученной модели, если модель уже была обучена

В методе update_measurements должно проводиться дообучение модели по вновь пришедшим данным

В методе get_frecast должен подготавливаться прогноз на следующие сутки и возвращаться в виде массива значений

"""
from flask import jsonify
import json
from forecaster import Forecaster
import pandas as pd
import random


class SunForecast(object):
    def __init__(self):
        pass

    def init(self, parameters):
        # Загрузка данных модели исходя из настроек в объекте parameters
        #
        self.files_container_list = parameters.files.getlist("files[]")
        self.filenames_list = []
        [self.filenames_list.append(item.filename) for item in self.files_container_list]
        
        self.jparameters = json.loads(parameters.form.to_dict(flat=False)["parameters"][0]) #чтение json из form-data
    
        self.X_scaler_filename = self.jparameters['files_destinations']['X_scaler_params']
        self.files_container_list[self.filenames_list.index('X_scaler_params.sca')].save(self.X_scaler_filename)

        self.y_scaler_filename = self.jparameters['files_destinations']['y_scaler_params']
        self.files_container_list[self.filenames_list.index('y_scaler_params.sca')].save(self.y_scaler_filename)

        self.training_dataset_filename = self.jparameters['files_destinations']['training_dataset']
        self.files_container_list[self.filenames_list.index('training_dataset_nvchb.csv')].save(self.training_dataset_filename)
        self.training_dataset_file_itself = pd.read_csv(self.training_dataset_filename, parse_dates=True, index_col='Unnamed: 0')
        
        self.updating_dataset_filename = self.jparameters['files_destinations']['updating_dataset']
        self.files_container_list[self.filenames_list.index('updating_dataset_nvchb.csv')].save(self.updating_dataset_filename)


        self.train_dataset_predictors = self.jparameters['training']['predictors']
        self.train_dataset_predictand = self.jparameters['training']['predictand']
        self.new_X_scaler_output_filename = self.jparameters['files_destinations']['new_X_scaler_filename']
        self.new_y_scaler_output_filename = self.jparameters['files_destinations']['new_y_scaler_filename']
        self.new_trained_model_filename = self.jparameters['files_destinations']['new_trained_model_filename']
        self.location_lat = float(self.jparameters['geography']['latitude'])
        self.location_lon = float(self.jparameters['geography']['longitude'])
        self.location_tz = self.jparameters['geography']['timezone']
        self.pv_tilt_angle = float(self.jparameters['pv_array']['pv_tilt_angle'])
        self.pv_azimuth_angle = float(self.jparameters['pv_array']['pv_azimuth_angle'])
        self.pv_modules_number = int(self.jparameters['pv_array']['pv_modules_number'])
        self.pv_model = self.jparameters['pv_array']['pv_module_params']

        #имена загружаемых файлов
        upl_trained_model_filename = 'trained_model.mdl'
        upl_training_dataset_filename = 'training_dataset_nvchb.csv'

        if upl_trained_model_filename in self.filenames_list: #если файл обученной модели загружен через post

            self.trained_model_filename = self.jparameters['files_destinations']['trained_model']
            self.files_container_list[self.filenames_list.index('trained_model.mdl')].save(self.trained_model_filename)

            return jsonify({"message": "ok", "trained_model_status": "file exists"})
        elif upl_training_dataset_filename in self.filenames_list: #если файл обучающих данных загружен через post
            try:
                Forecaster.train_new_model(Forecaster, self.training_dataset_filename, self.train_dataset_predictors, self.train_dataset_predictand, 
                    self.new_X_scaler_output_filename, self.new_y_scaler_output_filename, self.new_trained_model_filename)
                self.trained_model_filename = self.new_trained_model_filename
                self.X_scaler_file = self.new_X_scaler_output_filename
                self.y_scaler_file = self.new_y_scaler_output_filename
                return jsonify({"message": "ok", "trained_model_file": "no trained model file",
                "train_dataset": "loaded", "training": "new model has been trained and saved"})
            except:
                return jsonify({"message": "ok", "trained_model_file": "no trained model file",
            "train_dataset": "loaded", "training": "failed_to_train_new_model"})
        elif upl_training_dataset_filename not in self.filenames_list: #если не загружены обученная модель и обучающие данные
            return jsonify({"message": "ok", "trained_model_file": "no trained model file",
            "train_dataset": "no data"})
            
    def update_measurements(self, new_data):
        # Добавление вновь полученных измерений в модель прогнозирования
        data = json.loads(new_data.to_dict(flat=False)["data"][0])
        Forecaster.add_measurements_data(Forecaster,data['DateTime'], self.location_lat, self.location_lon, self.pv_tilt_angle,
        self.pv_azimuth_angle, [data['DateTime'], data['measurements']['T'], data['measurements']['U'], data['measurements']['Cl'], data['measurements']['I']], self.updating_dataset_filename)


    def get_forecast(self):
        # метод должен возвращать прогноз на сутки вперед
        # загрузка метеопрогноза с ресурса yr.no по академическому useragent
        weather_forecast = Forecaster.weather_fcst = Forecaster.yrnoparser(self.location_lat, self.location_lon, self.location_tz,  horizon=24)
        # прогнозирование интенсивности солнечной радиации
        I = Forecaster.mkforecast(self.trained_model_filename, weather_forecast,
                                       ['aoi', 'air_temperature', 'relative_humidity', 'cloud_area_fraction'],
                                       self.X_scaler_filename, self.y_scaler_filename,
                                       self.pv_tilt_angle, self.pv_azimuth_angle, test_mode=False).tz_localize(None)
        # вычисление выработки массива ФЭМ по спрогнозированной интенсивности солнечной радиации
        PV_output = Forecaster.PV_array_yield(self.pv_model['stc_power'], self.pv_model['area'], self.pv_model['kt'], I['irradiance'], I['air_temperature'], self.pv_modules_number)

        return I['irradiance'].to_json(date_format='iso', date_unit='s'), PV_output.to_json(date_format='iso', date_unit='s')

    def get_test_forecast(self):
        # тестовый прогноз на сутки 
        # входные данные за случайные последовательные 24 часа
        nrows = range(self.training_dataset_file_itself.shape[0])
        idx = random.randint(nrows.start, nrows.stop-24)
        # прогнозирование интенсивности солнечной радиации
        I = Forecaster.mkforecast(self.trained_model_filename, self.training_dataset_file_itself.iloc[idx:idx+24, :],
                                        ['aoi', 'T', 'U', 'Сl'],
                                        self.X_scaler_filename, self.y_scaler_filename,
                                        self.pv_tilt_angle, self.pv_azimuth_angle, test_mode=True).tz_localize(None)
        # вычисление выработки массива ФЭМ по спрогнозированной интенсивности солнечной радиации
        PV_output = Forecaster.PV_array_yield(self.pv_model['stc_power'], self.pv_model['area'], self.pv_model['kt'], I['irradiance'], I['T'], self.pv_modules_number)
      
        return I['irradiance'].to_json(date_format='iso', date_unit='s'), PV_output.to_json(date_format='iso', date_unit='s')