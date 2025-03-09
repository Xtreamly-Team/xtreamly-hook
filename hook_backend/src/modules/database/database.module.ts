import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { databaseConfig } from '@config/database.config';
import { User } from '../user/entities/user.entity/user.entity';
import { Position } from '../position/entities/position.entity/position.entity';

@Module({
  imports: [
    TypeOrmModule.forRootAsync({
      useFactory: () => databaseConfig,
    }),
  ],
})
export class DatabaseModule {}