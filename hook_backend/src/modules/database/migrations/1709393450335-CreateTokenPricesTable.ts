import { MigrationInterface, QueryRunner } from 'typeorm';
import { Client } from 'pg';
import { config } from 'dotenv';

// Load environment variables
config();

export class CreateTokenPricesTable1709393450335 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    // We need to bypass TypeORM's transaction handling completely
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

      // Create continuous aggregate view
      await client.query(`
        CREATE MATERIALIZED VIEW token_prices_hourly
        WITH (timescaledb.continuous) AS
        SELECT
          time_bucket('1 hour', timestamp) AS bucket,
          "tokenSymbol",
          "chainId",
          AVG("priceUsd") AS avg_price,
          MIN("priceUsd") AS min_price,
          MAX("priceUsd") AS max_price,
          FIRST("priceUsd", timestamp) AS open_price,
          LAST("priceUsd", timestamp) AS close_price,
          COUNT(*) AS sample_count
        FROM token_prices
        GROUP BY bucket, "tokenSymbol", "chainId";
      `);
      console.log('Created continuous aggregate view');

      // Check if continuous aggregate policy function exists
      const continuousAggregatePolicyResult = await client.query(`
        SELECT 1 FROM pg_proc WHERE proname = 'add_continuous_aggregate_policy';
      `);
      
      if (continuousAggregatePolicyResult.rows.length > 0) {
        await client.query(`
          SELECT add_continuous_aggregate_policy('token_prices_hourly',
            start_offset => INTERVAL '3 days',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
        `);
        console.log('Added continuous aggregate policy');
      } else {
        console.log('add_continuous_aggregate_policy function not found, skipping policy creation');
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

  public async down(queryRunner: QueryRunner): Promise<void> {
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
      console.log('Connected directly to database for rollback operations');

      // Check if continuous aggregate policy function exists
      const continuousAggregatePolicyResult = await client.query(`
        SELECT 1 FROM pg_proc WHERE proname = 'remove_continuous_aggregate_policy';
      `);
      
      if (continuousAggregatePolicyResult.rows.length > 0) {
        await client.query(`
          SELECT remove_continuous_aggregate_policy('token_prices_hourly', if_exists => TRUE);
        `);
        console.log('Removed continuous aggregate policy');
      }

      // Check if retention policy function exists
      const retentionPolicyResult = await client.query(`
        SELECT 1 FROM pg_proc WHERE proname = 'remove_retention_policy';
      `);
      
      if (retentionPolicyResult.rows.length > 0) {
        await client.query(`
          SELECT remove_retention_policy('token_prices', if_exists => TRUE);
        `);
        console.log('Removed retention policy');
      }

      // Drop materialized view
      await client.query('DROP MATERIALIZED VIEW IF EXISTS token_prices_hourly CASCADE;');
      console.log('Dropped materialized view');

      // Drop table and type
      await client.query('DROP TABLE IF EXISTS token_prices CASCADE;');
      await client.query('DROP TYPE IF EXISTS price_source_enum;');
      console.log('Dropped table and enum type');
      
      console.log('Rollback completed successfully');
    } catch (error) {
      console.error('Rollback failed:', error);
      throw error;
    } finally {
      // Close the direct connection
      await client.end();
      console.log('Closed direct database connection');
    }
  }
} 