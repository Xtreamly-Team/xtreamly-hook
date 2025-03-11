import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule } from '@nestjs/config';
import { ScheduleModule } from '@nestjs/schedule';

import { TokenPrice } from './entities/token-price.entity';
import { PriceRepository } from './repositories/price.repository';
import { PriceService } from './services/price.service';
import { PriceUpdateService } from './services/price-update.service';
import { CoinGeckoModule } from '../external-data/coin-gecko/coin-gecko.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([TokenPrice]),
    ConfigModule,
    ScheduleModule.forRoot(),
    CoinGeckoModule
  ],
  providers: [
    PriceRepository,
    PriceService,
    PriceUpdateService
  ],
  exports: [PriceService]
})
export class PriceModule {} 