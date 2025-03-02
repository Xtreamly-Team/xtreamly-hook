import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { env } from '@app/config/env.config';
import { User } from '../user/entities/user.entity/user.entity';
import { Position } from '../position/entities/position.entity/position.entity';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: env.DB_HOST,
      port: env.DB_PORT,
      username: env.DB_USERNAME,
      password: env.DB_PASSWORD,
      database: env.DB_NAME,
      entities: [User, Position],
      autoLoadEntities: true,
      synchronize: false,
      logging: env.NODE_ENV === 'development',
    }),
  ],
})
export class DatabaseModule {}