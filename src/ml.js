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
    console.log('Input tensor shape:', inputTensor.shape);
    console.log('Input tensor data:', await inputTensor.data());

    // Make prediction
    const prediction = model.predict(inputTensor);
    console.log('Prediction tensor shape:', prediction.shape);
    const predictionData = await prediction.data();
    console.log('Prediction data:', predictionData);

    // Clean up tensors
    inputTensor.dispose();
    prediction.dispose();

    const probability = predictionData[0];
    console.log('Raw probability:', probability);

    // Handle null or undefined probability
    if (probability === null || probability === undefined || isNaN(probability)) {
      console.error('Invalid probability value:', probability);
      console.log('Using fallback probability calculation...');

      // Fallback: calculate probability based on user data
      const fitnessScore = calculateFitnessScore(data);
      const fallbackProbability = Math.max(0.1, Math.min(0.9, fitnessScore / 100));
      console.log('Fallback probability:', fallbackProbability);

      const predictedClass = fallbackProbability >= 0.5 ? 'A' : 'B';
      const recommendations = getRecommendations(predictedClass, data, fallbackProbability);

      return {
        probability: fallbackProbability,
        class: predictedClass,
        recommendations: recommendations
      };
    }

    // Ensure probability is within valid range (0-1)
    const safeProbability = Math.max(0, Math.min(1, probability));
    console.log('Safe probability:', safeProbability);

    const predictedClass = safeProbability >= 0.5 ? 'A' : 'B';

    // Generate recommendations based on class
    const recommendations = getRecommendations(predictedClass, data, safeProbability);

    return {
      probability: safeProbability,
      class: predictedClass,
      recommendations: recommendations
    };
  } catch (error) {
    console.error('Prediction error:', error);

    // Fallback in case of model error
    console.log('Using emergency fallback...');
    const fitnessScore = calculateFitnessScore(data);
    const fallbackProbability = Math.max(0.1, Math.min(0.9, fitnessScore / 100));
    const predictedClass = fallbackProbability >= 0.5 ? 'A' : 'B';
    const recommendations = getRecommendations(predictedClass, data, fallbackProbability);

    return {
      probability: fallbackProbability,
      class: predictedClass,
      recommendations: recommendations
    };
  }
}

function getRecommendations(predictedClass, userData, probability) {
  // Calculate fitness score based on user characteristics
  const fitnessScore = calculateFitnessScore(userData);

  // Determine confidence level based on probability
  const confidence = getConfidenceLevel(probability);

  // Generate dynamic recommendations based on probability and user data
  const recommendations = generateDynamicRecommendations(userData, probability, fitnessScore);

  return {
    class: predictedClass,
    description: generateDescription(predictedClass, probability, fitnessScore),
    recommendations: recommendations.exercises,
    nutrition: recommendations.nutrition,
    goals: recommendations.goals,
    confidence: confidence,
    fitness_score: fitnessScore
  };
}

function calculateFitnessScore(userData) {
  // Calculate fitness score based on multiple factors
  let score = 0;

  // Age factor (optimal range 20-40)
  const ageScore = Math.max(0, 1 - Math.abs(userData.age - 30) / 30);
  score += ageScore * 0.2;

  // Height-weight ratio (BMI consideration)
  const heightM = userData.height_cm / 100;
  const bmi = userData.weight_kg / (heightM * heightM);
  const bmiScore = bmi >= 18.5 && bmi <= 25 ? 1 : Math.max(0, 1 - Math.abs(bmi - 21.75) / 10);
  score += bmiScore * 0.2;

  // Situps performance (normalized to 0-1)
  const situpsScore = Math.min(1, userData.situps_count / 50);
  score += situpsScore * 0.3;

  // Broad jump performance (normalized to 0-1)
  const jumpScore = Math.min(1, userData.broad_jump_cm / 300);
  score += jumpScore * 0.3;

  return Math.round(score * 100) / 100;
}

function getConfidenceLevel(probability) {
  if (probability >= 0.8 || probability <= 0.2) return 'High';
  if (probability >= 0.6 || probability <= 0.4) return 'Medium';
  return 'Low';
}

function generateDescription(predictedClass, probability, fitnessScore) {
  const confidence = getConfidenceLevel(probability);

  if (predictedClass === 'A') {
    if (confidence === 'High') {
      return `Berdasarkan analisis data Anda, Anda memiliki tingkat kebugaran yang sangat baik (skor: ${fitnessScore}/100). Model kami yakin dengan tingkat kepercayaan tinggi bahwa Anda siap untuk program latihan tingkat lanjut.`;
    } else if (confidence === 'Medium') {
      return `Analisis menunjukkan Anda memiliki potensi untuk program latihan tingkat lanjut (skor: ${fitnessScore}/100), namun disarankan untuk memulai dengan intensitas sedang dan meningkat secara bertahap.`;
    } else {
      return `Hasil analisis menunjukkan Anda berada di batas antara tingkat pemula dan lanjutan (skor: ${fitnessScore}/100). Disarankan untuk konsultasi dengan trainer untuk program yang tepat.`;
    }
  } else {
    if (confidence === 'High') {
      return `Berdasarkan analisis data Anda, Anda berada di tahap awal perjalanan fitness (skor: ${fitnessScore}/100). Model kami merekomendasikan program latihan dasar yang fokus pada membangun fondasi yang kuat.`;
    } else if (confidence === 'Medium') {
      return `Analisis menunjukkan Anda memiliki potensi untuk berkembang (skor: ${fitnessScore}/100). Mulailah dengan program latihan dasar dan tingkatkan intensitas secara bertahap.`;
    } else {
      return `Hasil analisis menunjukkan Anda berada di batas antara tingkat pemula dan lanjutan (skor: ${fitnessScore}/100). Disarankan untuk konsultasi dengan trainer untuk program yang tepat.`;
    }
  }
}

function generateDynamicRecommendations(userData, probability, fitnessScore) {
  const predictedClass = probability >= 0.5 ? 'A' : 'B';
  const confidence = getConfidenceLevel(probability);

  // Base recommendations
  let exercises = [];
  let nutrition = [];
  let goals = [];

  if (predictedClass === 'A') {
    // Advanced recommendations with dynamic intensity
    const intensity = Math.min(1, fitnessScore / 80);

    exercises = [
      `Latihan kardio intensitas tinggi (HIIT) ${Math.round(3 + intensity * 2)} kali seminggu`,
      `Latihan kekuatan dengan beban berat ${Math.round(4 + intensity)} kali seminggu`,
      'Fokus pada compound exercises (squat, deadlift, bench press)',
      `Latihan interval training ${Math.round(20 + intensity * 10)}-${Math.round(30 + intensity * 15)} menit`,
      confidence === 'High' ? 'Pertimbangkan untuk mengikuti kompetisi atau event fitness' : 'Konsultasi dengan personal trainer untuk program khusus',
      `Target peningkatan performa ${Math.round(10 + intensity * 15)}% dalam 3 bulan`
    ];

    nutrition = [
      `Asupan protein tinggi (${(1.6 + intensity * 0.6).toFixed(1)}-${(2.2 + intensity * 0.3).toFixed(1)}g per kg berat badan)`,
      'Karbohidrat kompleks untuk energi',
      intensity > 0.7 ? 'Suplemen kreatin untuk performa' : 'Suplemen protein whey',
      `Hidrasi yang cukup (${Math.round(3 + intensity)}-${Math.round(4 + intensity)} liter per hari)`
    ];

    goals = [
      `Meningkatkan kekuatan maksimal ${Math.round(15 + intensity * 20)}%`,
      `Mengurangi waktu latihan dengan intensitas tinggi ${Math.round(10 + intensity * 15)}%`,
      'Mencapai target performa spesifik',
      confidence === 'High' ? 'Mempersiapkan kompetisi atau event' : 'Membangun konsistensi latihan'
    ];
  } else {
    // Beginner recommendations with dynamic progression
    const progression = Math.min(1, (100 - fitnessScore) / 50);

    exercises = [
      `Mulai dengan latihan kardio ringan ${Math.round(2 + progression)}-${Math.round(3 + progression)} kali seminggu`,
      `Latihan kekuatan dengan beban ringan ${Math.round(2 + progression)} kali seminggu`,
      'Fokus pada teknik yang benar sebelum menambah intensitas',
      'Latihan bodyweight exercises (push-up, squat, plank)',
      'Konsultasi dengan trainer untuk program yang aman',
      `Mulai dengan durasi latihan ${Math.round(20 + progression * 10)}-${Math.round(30 + progression * 15)} menit`
    ];

    nutrition = [
      `Asupan protein moderat (${(1.2 + progression * 0.2).toFixed(1)}-${(1.6 + progression * 0.2).toFixed(1)}g per kg berat badan)`,
      'Karbohidrat seimbang untuk energi',
      'Vitamin dan mineral yang cukup',
      `Hidrasi yang cukup (${Math.round(2 + progression)}-${Math.round(3 + progression)} liter per hari)`
    ];

    goals = [
      'Membangun kebiasaan latihan rutin',
      `Meningkatkan stamina dasar ${Math.round(20 + progression * 30)}%`,
      'Mempelajari teknik latihan yang benar',
      `Mencapai target berat badan ideal dalam ${Math.round(6 + progression * 6)} bulan`
    ];
  }

  return { exercises, nutrition, goals };
}

module.exports = { loadModel, predict };