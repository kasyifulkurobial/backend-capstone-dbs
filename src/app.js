const express = require('express');
const cors = require('cors');
const path = require('path');
const { body, validationResult } = require('express-validator');
const { createClient } = require('@supabase/supabase-js');
const { loadModel, predict } = require('./ml');
require('dotenv').config();

// Initialize Express app
const app = express();

// Initialize Supabase client with service role key
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY, // Using service role key instead of anon key
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  }
);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// Serve documentation
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Validation middleware with strict ranges
const validatePredictionInput = [
  body('name').isString().notEmpty().trim().withMessage('Name is required'),
  body('age')
    .isInt({ min: 10, max: 100 })
    .withMessage('Age must be between 10 and 100 years'),
  body('height_cm')
    .isFloat({ min: 50, max: 250 })
    .withMessage('Height must be between 50 and 250 cm'),
  body('weight_kg')
    .isFloat({ min: 3, max: 300 })
    .withMessage('Weight must be between 3 and 300 kg'),
  body('situps_count')
    .isInt({ min: 0, max: 200 })
    .withMessage('Situps count must be between 0 and 200'),
  body('broad_jump_cm')
    .isFloat({ min: 0, max: 400 })
    .withMessage('Broad jump must be between 0 and 400 cm'),
];

// Routes
app.post('/predict', validatePredictionInput, async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array()
      });
    }

    const inputData = {
      name: req.body.name,
      age: req.body.age,
      height_cm: req.body.height_cm,
      weight_kg: req.body.weight_kg,
      situps_count: req.body.situps_count,
      broad_jump_cm: req.body.broad_jump_cm
    };

    // Get prediction
    const prediction = await predict(inputData);

    console.log('Prediction structure:', JSON.stringify(prediction, null, 2));

    // Validate prediction
    if (!prediction.probability || prediction.probability === null || isNaN(prediction.probability)) {
      throw new Error('Invalid prediction result: probability is null or invalid');
    }

    // Ensure probability doesn't exceed database constraint (max 0.9999 for numeric(4,4))
    const safeProbability = Math.min(0.9999, Math.max(0.0001, prediction.probability));

    // Store prediction in Supabase
    const { data, error } = await supabase
      .from('gym_predictions')
      .insert({
        ...inputData,
        predicted_class: prediction.class,
        probability: Number(safeProbability.toFixed(4)),
        description: prediction.recommendations.description,
        exercises: prediction.recommendations.recommendations,
        nutrition: prediction.recommendations.nutrition,
        goals: prediction.recommendations.goals
      })
      .select()
      .single();

    if (error) throw error;

    res.json({
      success: true,
      prediction: {
        class: prediction.class,
        probability: Number(safeProbability.toFixed(4)),
        description: prediction.recommendations.description,
        confidence: prediction.recommendations.confidence,
        fitness_score: prediction.recommendations.fitness_score
      },
      recommendations: {
        exercises: prediction.recommendations.recommendations,
        nutrition: prediction.recommendations.nutrition,
        goals: prediction.recommendations.goals
      },
      record: data
    });

  } catch (error) {
    console.error('Error processing prediction:', error);
    res.status(500).json({
      success: false,
      error: 'Error processing prediction'
    });
  }
});

// Get prediction history
app.get('/predictions', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('gym_predictions')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(100);

    if (error) throw error;

    res.json({
      success: true,
      predictions: data
    });

  } catch (error) {
    console.error('Error fetching predictions:', error);
    res.status(500).json({
      success: false,
      error: 'Error fetching predictions'
    });
  }
});

// Get prediction by ID
app.get('/predictions/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const { data, error } = await supabase
      .from('gym_predictions')
      .select('*')
      .eq('id', id)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return res.status(404).json({
          success: false,
          error: 'Prediction not found'
        });
      }
      throw error;
    }

    res.json({
      success: true,
      prediction: data
    });

  } catch (error) {
    console.error('Error fetching prediction:', error);
    res.status(500).json({
      success: false,
      error: 'Error fetching prediction'
    });
  }
});

// Initialize server
const PORT = process.env.PORT || 3000;

(async () => {
  try {
    // Load ML model
    await loadModel();

    // Start server
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
})();
