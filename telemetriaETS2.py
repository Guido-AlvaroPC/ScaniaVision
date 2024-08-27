import requests
import json
import time

# URL do servidor de telemetria
url = 'http://172.16.53.11:25555/api/ets2/telemetry'

def get_telemetry_data():
    try:
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        print(f"Erro ao acessar os dados de telemetria: {e}")
        return None

while True:
    telemetry_data = get_telemetry_data()
    if telemetry_data:
        truck_data = telemetry_data.get('truck', {})
        truck_data1 = telemetry_data.get('navigation', {})
        rpm = truck_data.get('engineRpm')                       # RPM
        speed = truck_data.get('speed')                         # Velocidade em Km/h
        odometer = truck_data.get('odometer')                   # Odometro
        gear = truck_data.get('gear')                           # Marcha
        fuel = truck_data.get('fuel')                           # Nivel de Combustivel
        fuelCapacity = truck_data.get('fuelCapacity')           # Capacidade max do Combustivel
        brakeTemperature = truck_data.get('brakeTemperature')   # Temperatura do Freio
        airPressure = truck_data.get('airPressure')             # Pressão no tanque de ar do freio em psi
        oilTemperature = truck_data.get('oilTemperature')       # Temperatura do óleo em graus Celsius
        oilPressure = truck_data.get('oilPressure')             # Pressão do óleo em psi
        waterTemperature = truck_data.get('waterTemperature')   # Temperatura da água em graus Celsius
        batteryVoltage = truck_data.get('batteryVoltage')       # Voltagem da bateria em volts
        speedLimit = truck_data1.get('speedLimit')              # Valor atual do "limite de velocidade do Route Advisor" em km/h

        print(f'''
              RPM: {rpm} 
              Velocidade: {speed:.2f} km/h 
              Marcha: {gear} 
              Radar: {speedLimit}
              ''')
    time.sleep(1)