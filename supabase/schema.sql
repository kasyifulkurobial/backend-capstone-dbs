-- Create the gym_predictions table
CREATE TABLE gym_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL,
    height_cm NUMERIC(5,2) NOT NULL,
    weight_kg NUMERIC(5,2) NOT NULL,
    situps_count INTEGER NOT NULL,
    broad_jump_cm NUMERIC(5,2) NOT NULL,
    predicted_class TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on created_at for better query performance
CREATE INDEX idx_gym_predictions_created_at ON gym_predictions(created_at);

-- Enable Row Level Security
ALTER TABLE gym_predictions ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if any
DROP POLICY IF EXISTS "Allow public access" ON gym_predictions;

-- Create a policy that allows public access
CREATE POLICY "Allow public access" ON gym_predictions
    FOR ALL
    TO PUBLIC
    USING (true)
    WITH CHECK (true); 