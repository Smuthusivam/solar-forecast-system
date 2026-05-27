import psycopg

# Step 1: Create the test database if it doesn't exist
try:
    with psycopg.connect("postgresql://postgres:Hjk125@localhost:5432/postgres", autocommit=True) as conn:
        conn.execute("CREATE DATABASE solar_forecast_test")
        print("Created solar_forecast_test database")
except psycopg.errors.DuplicateDatabase:
    print("solar_forecast_test already exists — skipping creation")

# Step 2: Grant schema permissions on the test database
with psycopg.connect("postgresql://postgres:Hjk125@localhost:5432/solar_forecast_test", autocommit=True) as conn:
    conn.execute("GRANT ALL ON SCHEMA public TO postgres")
    conn.execute("ALTER DATABASE solar_forecast_test OWNER TO postgres")
    print("Granted ALL on schema public to postgres")
    print("Done — run your tests now")
