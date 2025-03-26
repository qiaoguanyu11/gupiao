CREATE TABLE IF NOT EXISTS delete_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(10) NOT NULL,
    name VARCHAR(50) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    cost DECIMAL(10, 2) NOT NULL,
    quantity INT NOT NULL,
    total_cost DECIMAL(10, 2) NOT NULL,
    delete_time DATETIME NOT NULL,
    INDEX idx_code (code),
    INDEX idx_delete_time (delete_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4; 