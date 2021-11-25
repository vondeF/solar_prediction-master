from flask import Flask, abort, jsonify, app
from flask import request
from forecasting import SunForecast



"""
Точка входа в программу.
Запускается http-сервер для обработки REST и HTTP запросов
Создается объект модуля прогнозирования
"""

sun_forecast = SunForecast()
flask_app = Flask("ForecastAPI")

@flask_app.route('/init', methods=['POST'])
def init():
    try:
         #Инициализация модуля прогнозирования (в request.form лежат настройки для запуска)
        # например пути к файлам для обучения моделей или путь к параметрам обученной модели
        return sun_forecast.init(request)
    except:
        abort(500)

@flask_app.route('/update_measurements', methods=['POST'])
def update_measurements():
    try:
        #Добавление актуальных измерений в модель прогнозирования
        sun_forecast.update_measurements(request.form)
        return jsonify({"message": "ok"})
    except:
        abort(500)

@flask_app.route('/forecast', methods=['GET'])
def get_forecast():
    try:
        #Получение реального прогноза на сутки из модуля прогнозирования по данным от метеопровайдера
        return jsonify(sun_forecast.get_forecast())
    except:
        abort(500)

@flask_app.route('/test_forecast', methods=['GET'])
def get_test_forecast():
    try:
        #Получение тестового прогноза на сутки из модуля прогнозирования по архивным данным
        return jsonify(sun_forecast.get_test_forecast())
    except:
        abort(500)

if __name__=="__main__":
    flask_app.run(host='127.0.0.1', threaded=True, debug=True)
