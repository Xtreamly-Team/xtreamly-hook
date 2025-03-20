import { TypeOrmModuleOptions } from '@nestjs/typeorm';
import { env } from './env.config';
import { User } from '@modules/user/entities/user.entity/user.entity';
import { Position } from '@modules/position/entities/position.entity/position.entity';
import { PositionHistory } from '@modules/position/entities/position-history.entity/position-history.entity';

export const databaseConfig: TypeOrmModuleOptions = {
  type: 'postgres',
  host: env.DB_HOST,
  port: env.DB_HOST.startsWith('/cloudsql') ? undefined : env.DB_PORT,
  username: env.DB_USERNAME,
  password: env.DB_PASSWORD,
  database: env.DB_NAME,
  entities: [User, Position, PositionHistory],
  migrations: ['dist/modules/database/migrations/*{.ts,.js}'],
  autoLoadEntities: true,
  synchronize: false,
  logging: env.NODE_ENV === 'development',
  // Additional PostgreSQL specific configurations
  ssl: false, // env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
  extra: {
    max: 20, // connection pool max size
    connectionTimeoutMillis: 10000, // time to wait for connection
  },
};
