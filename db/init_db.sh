#!/bin/bash
# Sets up the local PostgreSQL database for this project.
# Run once before seeding: bash db/init_db.sh

set -e

DB_NAME="apollo_exercises"

echo "Dropping existing database if it exists..."
dropdb -U postgres --if-exists $DB_NAME

echo "Creating database..."
createdb -U postgres $DB_NAME

echo "Running schema..."
psql -U postgres -d $DB_NAME -f db/schema.sql

echo "Database ready."