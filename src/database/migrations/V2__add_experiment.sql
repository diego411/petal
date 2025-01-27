CREATE TABLE IF NOT EXISTS experiment (
     id SERIAL PRIMARY KEY,
     name TEXT NOT NULL,
     status TEXT NOT NULL,
     user_id INTEGER NOT NULL,
     created_at TIMESTAMP NOT NULL,
     started_at TIMESTAMP,
     recording_id INTEGER,
     FOREIGN KEY (recording_id) REFERENCES recording (id),
     FOREIGN KEY (user_id) REFERENCES users (id)
);
