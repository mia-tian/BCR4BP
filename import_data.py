import requests
import json
import numpy as np

def import_data(params):

    # Define the API endpoint
    api_url = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"


    # Make the request to the API
    response = requests.get(api_url, params=params)
    print('GET URL:', response.url)
    # response = requests.get('https://ssd-api.jpl.nasa.gov/periodic_orbits.api?sys=earth-moon&family=lyapunov&libr=1')

    # Check if the request was successful
    if response.status_code != 200:

        print(f"Error: Unable to fetch data (status code {response.status_code})")
        
    else:
        # Parse the JSON response
        data = response.json()

        return data

        # signature = data['signature']
        # family = data['family']
        # libration_point = data['libration_point']
        # # branch = data['branch']
        # limits = data['limits']
        # count = data['count']
        # fields = data['fields']
        # data = data['data']

        # data_matrix = np.matrix(data)
        # data_matrix = data_matrix.astype(float)

        # # Print the data (or process it as needed)
        # # print(json.dumps(data, indent=4))
        # return data
        # return data_matrix


