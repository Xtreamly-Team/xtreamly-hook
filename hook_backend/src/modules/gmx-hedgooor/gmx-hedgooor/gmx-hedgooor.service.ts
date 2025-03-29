import { Injectable, Logger } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import { PositionService } from '@modules/position/services/position/position.service';
import { UserService } from '@modules/user/services/user/user.service';
import { User } from '@modules/user/entities/user.entity/user.entity';
import { getPolicy } from '@modules/gmx-hedgooor/gmx-hedgooor/ai.service';
import { FirebaseUserService } from "@modules/firebase-user/services/firebase-user/firebase-user.service";
import { GMX } from "@app/utils/GMX";

@Injectable()
export class GmxHedgooorService {
  constructor(
    private readonly positionService: PositionService,
    private readonly userService: UserService,
    private readonly firebaseUserService: FirebaseUserService,
  ) {}

  private readonly logger = new Logger(GmxHedgooorService.name);

  @Cron('*/1 * * * *')
  async handleGmxQuery() {
    this.logger.log('Stared GMX Positions.');
    const users = await this.userService.getUsers();
    const hedgeCalls = users.map((user: User) => this.hedge(user));
    await Promise.all(hedgeCalls);
    this.logger.log('Ended GMX Positions.');
  }

  async hedge(user: User) {
    this.logger.log(`Starting hedging strategy for ${user.id}...`);
    const firebaseUser = await this.firebaseUserService.findByWalletAddress(user.walletAddress);
    const gmx = new GMX(firebaseUser.private);
    await gmx.init();

    const policy = await getPolicy(user.id);
    this.logger.log(`Policy for ${user.id}...`, policy);

    if (policy['perp_open']) {
      const position = await gmx.short(policy['collateral'], policy['leverage']);
      await this.positionService.createPosition(position);
    } else if (policy['perp_close']) {
      const position = await gmx.closeShort();
      await this.positionService.createPosition(position);
    }

    this.logger.log(`Ended hedging strategy for ${user.id}.`);
  }
}