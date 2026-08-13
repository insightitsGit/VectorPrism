-- VectorPrism / PSM 1024d tensor storage (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- Primary Storage Table for VectorPrism 1024d Tensors
CREATE TABLE IF NOT EXISTS psm_document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL UNIQUE,
    chunk_text TEXT NOT NULL,

    -- Full 1024-Dimensional Composite Tensor Payload
    tensor_1024d vector(1024) NOT NULL,

    -- Generated Column for Stage 1 Dense Core Slice [16..383] (368d)
    -- pgvector subvector() is 1-indexed: (17, 368) == zero-indexed [16:384)
    dense_core_slice vector(368) GENERATED ALWAYS AS (
        subvector(tensor_1024d, 17, 368)
    ) STORED,

    -- Header Metadata Fields (Extracted from Slice [0..15] at ingestion time)
    epistemic_truth FLOAT NOT NULL DEFAULT 1.0,
    anchor_dist FLOAT NOT NULL DEFAULT 0.0,
    valid_timestamp BIGINT NOT NULL,
    model_version INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- STAGE 1 INDEX: HNSW Index EXCLUSIVELY on 368d Dense Core Slice
CREATE INDEX IF NOT EXISTS idx_psm_dense_core_hnsw
ON psm_document_embeddings
USING hnsw (dense_core_slice vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Payload Indexes for Header Filtering
CREATE INDEX IF NOT EXISTS idx_psm_epistemic_truth ON psm_document_embeddings (epistemic_truth);
CREATE INDEX IF NOT EXISTS idx_psm_anchor_dist ON psm_document_embeddings (anchor_dist);
CREATE INDEX IF NOT EXISTS idx_psm_model_version ON psm_document_embeddings (model_version);
CREATE INDEX IF NOT EXISTS idx_psm_document_id ON psm_document_embeddings (document_id);
