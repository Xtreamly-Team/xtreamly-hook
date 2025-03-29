import { Module } from '@nestjs/common';
import { GmxHedgooorService } from './gmx-hedgooor/gmx-hedgooor.service';
import { PositionModule } from '@modules/position/position.module';
import { UserModule } from '@modules/user/user.module';
import { FirebaseUserService } from "@modules/firebase-user/services/firebase-user/firebase-user.service";

@Module({
  imports: [PositionModule, UserModule],
  providers: [GmxHedgooorService, FirebaseUserService],
  exports: [GmxHedgooorService],
})
export class GmxHedgooorModule {}