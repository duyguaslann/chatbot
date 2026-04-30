CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name  VARCHAR(100) NOT NULL,
    age        INTEGER,
    email      VARCHAR(255) UNIQUE NOT NULL,
    password   VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id         SERIAL PRIMARY KEY,
    user_id    VARCHAR(50) NOT NULL,
    title      VARCHAR(255) DEFAULT 'Yeni Sohbet',
    status     INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id           SERIAL PRIMARY KEY,
    user_type    INTEGER NOT NULL,
    message_text TEXT NOT NULL,
    chat_id      INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    status       INTEGER DEFAULT 1,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fine_tuning_data (
    id         SERIAL PRIMARY KEY,
    prompt     TEXT NOT NULL,
    completion TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS models (
    id         SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
