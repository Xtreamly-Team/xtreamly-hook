import { DataSource } from 'typeorm';
import { env } from './src/config/env.config';

export const dataSource = new DataSource({
  type: 'postgres',
  host: env.DB_HOST,
  port: env.DB_PORT,
  username: env.DB_USERNAME,
  password: env.DB_PASSWORD,
  database: env.DB_NAME,
  entities: ['dist/**/*.entity{.ts,.js}'],
  migrations: ['dist/modules/database/migrations/*{.ts,.js}'],
  logging: env.NODE_ENV === 'development',
  synchronize: false,
});