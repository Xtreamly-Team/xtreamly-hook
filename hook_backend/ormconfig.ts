import { DataSource } from 'typeorm';
import { config } from 'dotenv';
import { User } from './src/modules/user/entities/user.entity/user.entity';
import { Position } from './src/modules/position/entities/position.entity/position.entity';
import { PositionHistory } from './src/modules/position/entities/position-history.entity/position-history.entity';

// Load env variables
config();

export const dataSource = new DataSource({
  type: 'postgres',
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432'),
  username: process.env.DB_USERNAME || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME || 'xtr_trade_db',
  entities: [User, Position, PositionHistory],
  migrations: ['src/modules/database/migrations/*.ts'],
  migrationsTableName: 'migrations',
  logging: true,
  synchronize: false,
}); 