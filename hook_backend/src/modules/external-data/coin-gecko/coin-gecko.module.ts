import { Module } from '@nestjs/common';
import { CoinGeckoService } from './coin-gecko/coin-gecko.service';

@Module({
  providers: [CoinGeckoService]
})
export class CoinGeckoModule {}
