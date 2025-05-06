"""File used for making a local test request to the API."""

# call this file to test the API

import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "accommodates": 4,
    "bathrooms": 1,
    "bedrooms": 2,
    "beds": 2,
    "price": 100,
    "amenities_length": 15
}

response = requests.post(url, json=data)
print(response.json())

# or call the api from the command line using curl

'''
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "accommodates": 4,
           "bathrooms": 1,
           "bedrooms": 2,
           "beds": 2,
           "price": 100,
           "amenities_length": 15
         }'
'''
