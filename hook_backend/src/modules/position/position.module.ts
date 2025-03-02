import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Position } from './entities/position.entity/position.entity';
import { PositionHistory } from './entities/position-history.entity/position-history.entity';
import { PositionRepository } from './repositories/position.repository/position.repository';
import { PositionHistoryRepository } from './repositories/position-history.repository/position-history.repository';
import { PositionService } from './services/position/position.service';
import { PositionController } from '../../controllers/position/position.controller';
import { UserModule } from '../user/user.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([Position, PositionHistory]),
    UserModule,
  ],
  providers: [PositionRepository, PositionHistoryRepository, PositionService],
  controllers: [PositionController],
  exports: [PositionRepository, PositionHistoryRepository, PositionService],
})
export class PositionModule {}