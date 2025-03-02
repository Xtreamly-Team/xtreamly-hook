import { Module } from '@nestjs/common';
import { GmxHedgooorService } from './gmx-hedgooor/gmx-hedgooor.service';

@Module({
  providers: [GmxHedgooorService],
  exports: [GmxHedgooorService]
})
export class GmxHedgooorModule {}