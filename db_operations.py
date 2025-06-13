import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='YOURUSER',
            password='YOURPASSWORD',
            database='license_plate_db'
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def create_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='YOURUSER',
            password='YOURPASSWORD'
        )
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS license_plate_db")
        connection.close()
    except Error as e:
        print(f"Error creating database: {e}")

def create_table():
    try:
        connection = create_connection()
        if connection is None:
            return
        
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS license_plates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                plate_number VARCHAR(20) NOT NULL,
                owner_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        connection.close()
    except Error as e:
        print(f"Error creating table: {e}")

def insert_plate(plate_number, owner_name=None):
    try:
        connection = create_connection()
        if connection is None:
            return False
        
        cursor = connection.cursor()
        sql = "INSERT INTO license_plates (plate_number, owner_name) VALUES (%s, %s)"
        cursor.execute(sql, (plate_number, owner_name))
        connection.commit()
        connection.close()
        return True
    except Error as e:
        print(f"Error inserting record: {e}")
        return False

def get_all_plates():
    try:
        connection = create_connection()
        if connection is None:
            return []
        
        cursor = connection.cursor()
        cursor.execute("SELECT plate_number, owner_name FROM license_plates")
        records = cursor.fetchall()
        connection.close()
        return records
    except Error as e:
        print(f"Error fetching records: {e}")
        return [] 