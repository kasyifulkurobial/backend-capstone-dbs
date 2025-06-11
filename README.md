# Gym Class Prediction API

This project provides a REST API for predicting gym class assignments based on physical attributes and performance metrics. The API uses a TensorFlow.js model to classify users into class 'A' or 'B' based on their physical measurements and performance.

## Dependencies

- Express.js
- TensorFlow.js
- Supabase
- Express Validator
- CORS
- Dotenv

## Machine Learning Model

The model was trained to classify gym participants into two classes (A or B) based on the following features:

- Age
- Height
- Weight
- Situps Count
- Broad Jump Distance

The original model was created in Keras (.h5 format) and converted to TensorFlow.js format. To convert a Keras model to TensorFlow.js format, you can use the following methods:

### Using Python

```python
from tensorflow.keras.models import load_model
import tensorflowjs as tfjs

model_path = load_model("<Path to your model>")
output_path = "<Output path after your model is converted>"

tfjs.converters.save_keras_model(model_path, output_path)
```

### Using Command Line

```bash
tensorflowjs_converter --input_format=keras <path to your model> <output_path>
```

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
```

2. Install dependencies:

```bash
npm install
```

3. Create a `.env` file with the following variables:

```
PORT=3000
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
NODE_ENV=development
```

4. Run the development server:

```bash
npm run dev
```

## API Endpoints

### 1. Make Prediction

**Endpoint:** `POST /predict`

**Request Body:**

```json
{
    "name": "John Doe",
    "age": 25,
    "height_cm": 170,
    "weight_kg": 70,
    "situps_count": 20,
    "broad_jump_cm": 200
}
```

**Input Constraints:**

- name: Required string
- age: Integer between 10 and 100
- height_cm: Float between 50 and 250
- weight_kg: Float between 3 and 300
- situps_count: Integer between 0 and 200
- broad_jump_cm: Float between 0 and 400

**Success Response:**

```json
{
    "success": true,
    "prediction": {
        "class": "A",
        "probability": "75%"
    },
    "record": {
        "id": "uuid",
        "name": "John Doe",
        "age": 25,
        "height_cm": 170,
        "weight_kg": 70,
        "situps_count": 20,
        "broad_jump_cm": 200,
        "predicted_class": "A",
        "created_at": "2024-03-11T..."
    }
}
```

**Error Response:**

```json
{
    "success": false,
    "errors": [
        {
            "type": "field",
            "msg": "Age must be between 10 and 100 years",
            "path": "age",
            "location": "body"
        }
    ]
}
```

### 2. Get Prediction History

**Endpoint:** `GET /predictions`

**Response:**

```json
{
    "success": true,
    "predictions": [
        {
            "id": "uuid",
            "name": "John Doe",
            "age": 25,
            "height_cm": 170,
            "weight_kg": 70,
            "situps_count": 20,
            "broad_jump_cm": 200,
            "predicted_class": "A",
            "created_at": "2024-03-11T..."
        }
    ]
}
```

## Deployment to Google Cloud Compute Engine

1. Set firewall rule for port 3000

2. Create your instance and apply the firewall rule via [target tags](https://cloud.google.com/vpc/docs/add-remove-network-tags)

3. Connect to the instance via SSH

4. Install Node.js:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18
```

5. Clone and set up the project following the installation steps above

## Notes

- The API uses TensorFlow.js for model inference
- Predictions are stored in Supabase database
- The model classifies users into two classes: 'A' (probability >= 0.5) or 'B' (probability < 0.5)
- All measurements must be within the specified ranges to get a valid prediction
