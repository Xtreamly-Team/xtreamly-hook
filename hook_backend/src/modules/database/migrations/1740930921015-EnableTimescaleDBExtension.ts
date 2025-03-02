import { MigrationInterface, QueryRunner } from 'typeorm';

export class EnableTimescaleDBExtension1709393450333 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;`);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP EXTENSION IF EXISTS "timescaledb" CASCADE;`);
  }
}