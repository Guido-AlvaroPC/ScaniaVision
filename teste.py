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
        # Acessando os dados específicos de RPM e velocidade (km/h)
        truck_data = telemetry_data.get('truck', {})
        rpm = truck_data.get('engineRpm')
        speed = truck_data.get('speed', 0.0)  # Convertendo de m/s para km/h
        
        print(f"RPM: {rpm}, Velocidade: {speed:.2f} km/h")
    
    # Aguarde um pouco antes de fazer a próxima leitura
    time.sleep(1)