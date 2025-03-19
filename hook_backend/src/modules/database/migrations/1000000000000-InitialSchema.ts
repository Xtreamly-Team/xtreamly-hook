import { MigrationInterface, QueryRunner } from 'typeorm';

export class InitialSchema1000000000000 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    // Enable extensions
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS "uuid-ossp" CASCADE;`);
    // await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;`);

    // Create users table
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS "users" (
        "id" uuid DEFAULT uuid_generate_v4(),
        "walletAddress" varchar NOT NULL UNIQUE,
        "email" varchar NULL,
        "isActive" boolean DEFAULT true,
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY ("id")
      )
    `);

    // Create positions table
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS "positions" (
        "id" uuid DEFAULT uuid_generate_v4(),
        "userId" uuid NOT NULL,
        "tokenA" varchar NOT NULL,
        "tokenB" varchar NOT NULL,
        "amountA" decimal(24,8) NOT NULL,
        "amountB" decimal(24,8) NOT NULL,
        "lowerTick" decimal(24,8) NOT NULL,
        "upperTick" decimal(24,8) NOT NULL,
        "hedgeAmount" decimal(24,8) NULL,
        "status" varchar NOT NULL DEFAULT 'pending',
        "uniswapPositionId" varchar NULL,
        "gmxPositionId" varchar NULL,
        "metadata" jsonb NULL,
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY ("id"),
        FOREIGN KEY ("userId") REFERENCES "users"("id")
      )
    `);

    // Create position_history table
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS "position_history" (
        "id" uuid DEFAULT uuid_generate_v4(),
        "positionId" uuid NOT NULL,
        "tokenAValue" decimal(24,8) NOT NULL,
        "tokenBValue" decimal(24,8) NOT NULL,
        "hedgeValue" decimal(24,8) NOT NULL,
        "netValue" decimal(24,8) NOT NULL,
        "metadata" jsonb NULL,
        "timestamp" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY ("id", "timestamp"),
        FOREIGN KEY ("positionId") REFERENCES "positions"("id")
      )
    `);

    // Create indexes
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS "IDX_positions_userId" ON "positions"("userId");
      CREATE INDEX IF NOT EXISTS "IDX_positions_status" ON "positions"("status");
      CREATE INDEX IF NOT EXISTS "IDX_position_history_timestamp" ON "position_history"("timestamp" DESC);
      CREATE INDEX IF NOT EXISTS "IDX_position_history_positionId" ON "position_history"("positionId");
    `);

    // Convert position_history to hypertable
    // await queryRunner.query(`
    //   SELECT create_hypertable('position_history', 'timestamp',
    //     chunk_time_interval => interval '1 day',
    //     if_not_exists => TRUE
    //   );
    // `);

    // Add retention policy - keep data for 90 days
    // await queryRunner.query(`
    //   SELECT add_retention_policy('position_history', INTERVAL '90 days', if_not_exists => TRUE);
    // `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    // Remove retention policy
    // await queryRunner.query(`
    //   SELECT remove_retention_policy('position_history', if_not_exists => TRUE);
    // `);
    
    await queryRunner.query(`DROP TABLE IF EXISTS "position_history" CASCADE`);
    await queryRunner.query(`DROP TABLE IF EXISTS "positions" CASCADE`);
    await queryRunner.query(`DROP TABLE IF EXISTS "users" CASCADE`);
    
    // await queryRunner.query(`DROP EXTENSION IF EXISTS "timescaledb" CASCADE;`);
    await queryRunner.query(`DROP EXTENSION IF EXISTS "uuid-ossp" CASCADE;`);
  }
} 