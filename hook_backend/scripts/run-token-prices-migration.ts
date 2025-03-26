import { Client } from 'pg';
import { config } from 'dotenv';

// Load environment variables
config();

async function runMigration() {
  console.log('Starting migration for token_prices table and materialized view...');
  
  // Create a direct connection to the database
  const client = new Client({
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    user: process.env.DB_USERNAME || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    database: process.env.DB_NAME || 'xtr_trade_db',
  });

  try {
    // Connect directly to the database
    await client.connect();
    console.log('Connected directly to database for TimescaleDB operations');

    // Check if TimescaleDB extension is installed
    const extensionCheck = await client.query(`
      SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');
    `);
    
    if (!extensionCheck.rows[0].exists) {
      throw new Error('TimescaleDB extension is not installed. Please install it first.');
    }
    
    // Get TimescaleDB version
    const versionResult = await client.query(`
      SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'
    `);
    const timescaleVersion = versionResult.rows[0]?.extversion || 'unknown';
    console.log(`TimescaleDB extension is installed (version ${timescaleVersion})`);

    // Clean up any existing objects from failed migrations
    await client.query('DROP MATERIALIZED VIEW IF EXISTS token_prices_hourly CASCADE;');
    await client.query('DROP TABLE IF EXISTS token_prices CASCADE;');
    await client.query('DROP TYPE IF EXISTS price_source_enum;');
    
    console.log('Cleaned up any existing objects');

    // Create enum type
    await client.query(`
      CREATE TYPE price_source_enum AS ENUM ('coingecko', 'fallback', 'manual');
    `);
    console.log('Created price_source_enum type');

    // Create table
    await client.query(`
      CREATE TABLE token_prices (
        id uuid DEFAULT uuid_generate_v4(),
        "tokenSymbol" varchar NOT NULL,
        "tokenAddress" varchar NULL,
        "priceUsd" decimal(24,8) NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        source price_source_enum NOT NULL DEFAULT 'coingecko',
        "chainId" integer NOT NULL,
        metadata jsonb NULL,
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id, timestamp)
      );
    `);
    console.log('Created token_prices table');

    // Create indexes
    await client.query(`
      CREATE INDEX "IDX_token_prices_tokenSymbol" ON token_prices("tokenSymbol", timestamp DESC);
      CREATE INDEX "IDX_token_prices_timestamp" ON token_prices(timestamp DESC);
      CREATE INDEX "IDX_token_prices_chainId" ON token_prices("chainId", timestamp DESC);
      CREATE INDEX "IDX_token_prices_symbol_chain" ON token_prices("tokenSymbol", "chainId", timestamp DESC);
    `);
    console.log('Created indexes');

    // Convert to hypertable
    await client.query(`
      SELECT create_hypertable('token_prices', 'timestamp', 
        chunk_time_interval => interval '1 day',
        if_not_exists => TRUE
      );
    `);
    console.log('Converted to hypertable');

    // Try to add retention policy if the function exists
    try {
      // Check if retention policy function exists
      const retentionPolicyResult = await client.query(`
        SELECT 1 FROM pg_proc WHERE proname = 'add_retention_policy';
      `);
      
      if (retentionPolicyResult.rows.length > 0) {
        await client.query(`
          SELECT add_retention_policy('token_prices', INTERVAL '90 days', if_not_exists => TRUE);
        `);
        console.log('Added retention policy');
      } else {
        console.log('add_retention_policy function not found, skipping retention policy creation');
      }
    } catch (error) {
      console.log('Skipping retention policy - not supported in this TimescaleDB version');
    }

    // Create materialized view for hourly aggregation
    console.log('Creating materialized view for hourly aggregation');
    await client.query(`
      CREATE MATERIALIZED VIEW token_prices_hourly AS
      SELECT
        date_trunc('hour', timestamp) AS bucket,
        "tokenSymbol",
        "chainId",
        AVG("priceUsd"::numeric) AS avg_price,
        MIN("priceUsd"::numeric) AS min_price,
        MAX("priceUsd"::numeric) AS max_price,
        (array_agg("priceUsd" ORDER BY timestamp ASC))[1] AS open_price,
        (array_agg("priceUsd" ORDER BY timestamp DESC))[1] AS close_price,
        COUNT(*) AS sample_count
      FROM token_prices
      GROUP BY date_trunc('hour', timestamp), "tokenSymbol", "chainId";
    `);
    
    // Create an index on the view
    await client.query(`
      CREATE INDEX ON token_prices_hourly (bucket, "tokenSymbol", "chainId");
    `);
    
    console.log('Created materialized view with index');
    
    // Check if migrations table exists
    const migrationsTableExists = await client.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'migrations'
      );
    `);
    
    // Update migrations table to mark this migration as run
    if (migrationsTableExists.rows[0].exists) {
      await client.query(`
        INSERT INTO migrations (timestamp, name) 
        VALUES (1709393450335, 'CreateTokenPricesTable1709393450335')
        ON CONFLICT DO NOTHING;
      `);
      console.log('Updated migrations table');
    } else {
      console.log('Migrations table does not exist, skipping update');
    }
    
    console.log('Migration completed successfully');
  } catch (error) {
    console.error('Migration failed:', error);
    throw error;
  } finally {
    // Close the direct connection
    await client.end();
    console.log('Closed direct database connection');
  }
}

// Run the migration
runMigration()
  .then(() => {
    console.log('Migration script completed successfully');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Migration script failed:', error);
    process.exit(1);
  }); 