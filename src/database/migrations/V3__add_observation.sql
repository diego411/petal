CREATE TABLE IF NOT EXISTS observation (
    id SERIAL PRIMARY KEY,
    label TEXT NOT NULL,
    observed_at TIMESTAMP NOT NULL,
    experiment_id INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiment (id)
);
