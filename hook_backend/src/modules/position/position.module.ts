import { Module } from '@nestjs/common';
import { PositionService } from './services/position/position.service';

@Module({
  providers: [PositionService]
})
export class PositionModule {}
