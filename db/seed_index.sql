-- Build a vector similarity index after data has been loaded.
-- IVFFlat works by clustering vectors into groups (lists) at index time,
-- so it needs existing data to build from and this is why it runs after seeding.
CREATE INDEX ON exercises USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);