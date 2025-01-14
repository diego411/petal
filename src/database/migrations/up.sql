CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recording (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    state INTEGER NOT NULL,
    sample_rate INTEGER NOT NULL,
    threshold INTEGER NOT NULL,
    start_time TIMESTAMP,
    last_update TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS measurement (
    id SERIAL,
    value REAL,
    recording INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (recording) REFERENCES recording(id) ON DELETE CASCADE,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);
