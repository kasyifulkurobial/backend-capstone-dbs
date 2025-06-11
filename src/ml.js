const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

let model = null;

async function loadModel() {
  try {
    // Create sequential model with explicit input shape
    model = tf.sequential({
      layers: [
        tf.layers.inputLayer({ inputShape: [5] }),
        tf.layers.dense({
          units: 32,
          activation: 'relu'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'sigmoid'
        })
      ]
    });

    // Load weights
    const weightsPath = path.join(__dirname, '../models/group1-shard1of1.bin');
    const weightsBuffer = fs.readFileSync(weightsPath);
    const weights = new Float32Array(weightsBuffer.buffer);

    // Set weights for each layer
    let offset = 0;
    for (const layer of model.layers) {
      if (layer.weights.length > 0) { // Skip input layer
        const layerWeights = [];
        for (const weight of layer.weights) {
          const shape = weight.shape;
          const size = shape.reduce((a, b) => a * b);
          const values = weights.slice(offset, offset + size);
          const tensor = tf.tensor(values, shape);
          layerWeights.push(tensor);
          offset += size;
        }
        layer.setWeights(layerWeights);
      }
    }

    console.log('Model loaded successfully');
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

function preprocessInput(data) {
  // Create input array in the correct order
  const inputData = [
    data.age,
    data.height_cm,
    data.weight_kg,
    data.situps_count,
    data.broad_jump_cm
  ];

  // Normalize the input data
  const normalizedData = [
    (inputData[0] - 55) / 45, // age range 10 to 100
    (inputData[1] - 150) / 100, // height range 50 to 250 cm
    (inputData[2] - 151.5) / 148.5, // weight range 3 to 300 kg
    (inputData[3] - 100) / 100, // situps range 0 to 200
    (inputData[4] - 200) / 200 // broad jump range 0 to 400 cm
  ];

  return tf.tensor2d([normalizedData]);
}

async function predict(data) {
  if (!model) {
    throw new Error('Model not loaded');
  }

  try {
    // Preprocess the input data
    const inputTensor = preprocessInput(data);

    // Make prediction
    const prediction = model.predict(inputTensor);
    const predictionData = await prediction.data();

    // Clean up tensors
    inputTensor.dispose();
    prediction.dispose();

    return {
      probability: predictionData[0]
    };
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}

module.exports = { loadModel, predict };