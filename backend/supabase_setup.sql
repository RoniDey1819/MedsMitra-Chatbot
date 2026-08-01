-- Run this in Supabase → SQL Editor.
-- Safe to re-run: uses IF NOT EXISTS / CREATE OR REPLACE throughout, so it
-- also works as an upgrade script on a table created by the old version.

-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Table holding one row per medicine, plus its embedding and metadata.
-- Vector size = 384 because we use the "all-MiniLM-L6-v2" sentence-transformers
-- model. If you switch embedding models, update this number to match.
create table if not exists medicines (
    medicine_id         text primary key,
    medicine_name       text not null,
    strength            text,
    use_case            text,
    alternative         text,
    stock               text,
    dosage_instruction  text,
    content             text not null,   -- flattened text that was embedded
    embedding           vector(384)
);

-- 2a. Metadata columns used for filtering (added via ALTER so this script
-- also upgrades a pre-existing table without dropping data).
alter table medicines add column if not exists category text;
alter table medicines add column if not exists in_stock boolean;

comment on column medicines.category is
    'Coarse category derived from Use_Case, e.g. "Fever" — used for metadata filtering.';
comment on column medicines.in_stock is
    'Boolean form of the Stock column (Yes/No), used for metadata filtering.';

-- 3. Vector index for fast similarity search.
-- HNSW is the current recommended index type for pgvector cosine search
-- (better recall/latency than ivfflat and needs no "lists" tuning based on
-- row count). Drop any old ivfflat index from a previous version first.
drop index if exists medicines_embedding_idx;

create index if not exists medicines_embedding_hnsw_idx
    on medicines
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- Helpful indexes for the metadata filters used by match_medicines below.
create index if not exists medicines_category_idx on medicines (category);
create index if not exists medicines_in_stock_idx on medicines (in_stock);

-- 4. Function used by the backend to run similarity search via RPC.
-- Adds a similarity_threshold (rows below it are excluded, not just ranked
-- lower) and optional metadata filters for category / in_stock.
create or replace function match_medicines (
    query_embedding vector(384),
    match_count int default 5,
    similarity_threshold float default 0.25,
    filter_category text default null,
    filter_in_stock boolean default null
)
returns table (
    medicine_id         text,
    medicine_name       text,
    strength            text,
    use_case            text,
    alternative         text,
    stock               text,
    dosage_instruction  text,
    category            text,
    in_stock            boolean,
    content             text,
    similarity          float
)
language sql stable
as $$
    select
        medicine_id,
        medicine_name,
        strength,
        use_case,
        alternative,
        stock,
        dosage_instruction,
        category,
        in_stock,
        content,
        1 - (embedding <=> query_embedding) as similarity
    from medicines
    where
        (1 - (embedding <=> query_embedding)) >= similarity_threshold
        and (filter_category is null or category = filter_category)
        and (filter_in_stock is null or in_stock = filter_in_stock)
    order by embedding <=> query_embedding
    limit match_count;
$$;