import { Module } from '@nestjs/common';
import { GmxHedgooorService } from './gmx-hedgooor/gmx-hedgooor.service';

@Module({
  providers: [GmxHedgooorService]
})
export class GmxHedgooorModule {}
