-- 1. Core tables
CREATE TABLE concept (
  concept_id   BIGSERIAL PRIMARY KEY,
  name         TEXT        NOT NULL UNIQUE
);

CREATE TABLE app_user (
  user_id      BIGSERIAL PRIMARY KEY,
  username     TEXT        NOT NULL UNIQUE,
  email        TEXT        NOT NULL UNIQUE,
  is_active    BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE TABLE role (
  role_id      BIGSERIAL PRIMARY KEY,
  role_name    TEXT        NOT NULL UNIQUE
);

CREATE TABLE permission (
  permission_id   BIGSERIAL PRIMARY KEY,
  codename        TEXT        NOT NULL UNIQUE
);

CREATE TABLE user_role_concept (
  user_id     BIGINT  NOT NULL REFERENCES app_user(user_id),
  role_id     BIGINT  NOT NULL REFERENCES role(role_id),
  concept_id  BIGINT  NOT NULL REFERENCES concept(concept_id),
  PRIMARY KEY(user_id, role_id, concept_id)
);

CREATE TABLE role_permission (
  role_id        BIGINT  NOT NULL REFERENCES role(role_id),
  permission_id  BIGINT  NOT NULL REFERENCES permission(permission_id),
  PRIMARY KEY(role_id, permission_id)
);

-- 2. Business data + RAG metadata
CREATE TABLE sales_order (
  order_id      BIGSERIAL PRIMARY KEY,
  concept_id    BIGINT    NOT NULL REFERENCES concept(concept_id),
  order_date    DATE      NOT NULL,
  total_amount  NUMERIC(12,2) NOT NULL
);
ALTER TABLE sales_order ENABLE ROW LEVEL SECURITY;

CREATE TABLE rag_chunk (
  chunk_id      BIGSERIAL PRIMARY KEY,
  concept_id    BIGINT    NOT NULL REFERENCES concept(concept_id),
  table_name    TEXT      NOT NULL,
  row_pk        TEXT      NOT NULL,
  embedding     VECTOR(1536) NOT NULL
);
ALTER TABLE rag_chunk ENABLE ROW LEVEL SECURITY;

-- 3. RLS policies
CREATE POLICY sales_read_policy
  ON sales_order FOR SELECT
  USING (
    EXISTS (
      SELECT 1
      FROM user_role_concept urc
      JOIN role_permission rp ON urc.role_id = rp.role_id
      JOIN permission p       ON rp.permission_id = p.permission_id
      WHERE urc.user_id    = current_setting('app.current_user_id')::BIGINT
        AND urc.concept_id = sales_order.concept_id
        AND p.codename     = 'read_sales'
    )
  );

CREATE POLICY rag_chunk_read_policy
  ON rag_chunk FOR SELECT
  USING (
    EXISTS (
      SELECT 1
      FROM user_role_concept urc
      WHERE urc.user_id    = current_setting('app.current_user_id')::BIGINT
        AND urc.concept_id = rag_chunk.concept_id
    )
  );
