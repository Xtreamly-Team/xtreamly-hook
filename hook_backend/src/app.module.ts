import { Module } from '@nestjs/common';
import { DevtoolsModule } from '@nestjs/devtools-integration';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UserController } from './controllers/user/user.controller';
import { PositionController } from './controllers/position/position.controller';
import { QuoteComputeModule } from './modules/quote-compute/quote-compute.module';
import { UpdateUserPositionModule } from './modules/update-user-position/update-user-position.module';
import { XtrV4UniHookModule } from './modules/xtr_v4_uni_hook/xtr_v4_uni_hook.module';
import { GmxHedgooorModule } from './modules/gmx-hedgooor/gmx-hedgooor.module';
import { SharedModule } from './modules/shared/shared.module';
import { XtreamlyApiModule } from './modules/external-data/xtreamly-api/xtreamly-api.module';
import { CoinGeckoModule } from './modules/external-data/coin-gecko/coin-gecko.module';
import { DatabaseModule } from './modules/database/database.module';
import { UserModule } from './modules/user/user.module';
import { PositionModule } from './modules/position/position.module';

@Module({
  imports: [
    DevtoolsModule.register({
      http: process.env.NODE_ENV !== 'production',
    }),
    QuoteComputeModule,
    UpdateUserPositionModule,
    XtrV4UniHookModule,
    GmxHedgooorModule,
    SharedModule,
    XtreamlyApiModule,
    CoinGeckoModule,
    DatabaseModule,
    UserModule,
    PositionModule,
  ],
  controllers: [AppController, UserController, PositionController],
  providers: [AppService],
})
export class AppModule {}