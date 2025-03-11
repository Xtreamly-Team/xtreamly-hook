import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { CoinGeckoService } from './coin-gecko.service';

@Module({
  imports: [ConfigModule],
  providers: [CoinGeckoService],
  exports: [CoinGeckoService]
})
export class CoinGeckoModule {}
