import { Module } from '@nestjs/common';
import { UpdateUserPositionService } from './update-user-position/update-user-position.service';

@Module({
  providers: [UpdateUserPositionService]
})
export class UpdateUserPositionModule {}
