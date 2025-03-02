import { MigrationInterface, QueryRunner } from "typeorm";

export class CreatePositionHistoryHypertable1740940117272 implements MigrationInterface {

    public async up(queryRunner: QueryRunner): Promise<void> {
        // Ensure the table exists first before creating the hypertable
        await queryRunner.query(`
          SELECT create_hypertable('position_history', 'timestamp', 
            chunk_time_interval => interval '1 day',
            if_not_exists => TRUE
          );
        `);
    
        // Add retention policy - keep data for 90 days
        await queryRunner.query(`
          SELECT add_retention_policy('position_history', INTERVAL '90 days', if_not_exists => TRUE);
        `);
      }
    
      public async down(queryRunner: QueryRunner): Promise<void> {
        // Remove retention policy
        await queryRunner.query(`
          SELECT remove_retention_policy('position_history', if_not_exists => TRUE);
        `);
      }
    }