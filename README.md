# AirRate

A neural network that predicts Airbnb ratings based on several features.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Setup](#setup)
  - [Local Environment Setup](#local-environment-setup)
  - [Dependencies](#dependencies)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Model Deployment (API)](#model-deployment-api)
  - [Querying the Model](#querying-the-model)
- [License](#license)

## Description

AirRate is a machine learning project that uses a neural network to predict Airbnb ratings based on various features such as the number of bedrooms, bathrooms, price, and amenities. This project was developed as part of a final group project for the Deep Learning course at Allegheny College.

## Features

- Predict Airbnb ratings using a trained neural network.
- Preprocessing and feature scaling for input data.
- REST API for model deployment and querying.

## Setup

### Local Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/air-rate.git
   cd air-rate
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Set the Python version (if using `asdf`):

   ```bash
   asdf set python 3.10.12
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following Python libraries:

- `numpy<2.0`
- `pandas`
- `torch==2.0.1`
- `scikit-learn==1.3.0`
- `fastapi==0.95.2`
- `pydantic==1.10.7`
- `uvicorn==0.22.0`
- `requests`

## Usage

### Model Training

1. Navigate to the `src` directory:

   ```bash
   cd src
   ```

2. Train the model:

   ```bash
   python main.py
   ```

### Model Deployment (API)

1. Start the API server using Uvicorn:

   ```bash
   uvicorn api.api:app --reload --host 127.0.0.1 --port 8000
   ```

2. The API will be available at `http://127.0.0.1:8000`.

### Querying the Model

To query the model, send a POST request to the `/predict` endpoint. For example:

```bash
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
```

The API will return a JSON response with the predicted rating.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
