CREATE DATABASE license_plate_db;

CREATE TABLE IF NOT EXISTS license_plates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                plate_number VARCHAR(20) NOT NULL,
                owner_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )