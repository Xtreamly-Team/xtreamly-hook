import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { PriceRepository } from '../repositories/price.repository';
import { CoinGeckoService } from '../../external-data/coin-gecko/coin-gecko.service';
import { PriceSource } from '../entities/token-price.entity';

@Injectable()
export class PriceService {
  private readonly logger = new Logger(PriceService.name);
  private readonly supportedTokens: string[];
  private readonly chainId: number;

  constructor(
    private readonly priceRepository: PriceRepository,
    private readonly coinGeckoService: CoinGeckoService,
    private readonly configService: ConfigService,
  ) {
    this.supportedTokens = ['ETH', 'USDC'];
    this.chainId = this.configService.get<number>('CHAIN_ID', 42161); // Default to Arbitrum One
  }

  async updatePrices(): Promise<void> {
    try {
      const prices = await this.coinGeckoService.getMultipleTokenPrices(this.supportedTokens);
      
      for (const tokenPrice of prices) {
        await this.priceRepository.savePrice(
          tokenPrice.symbol,
          tokenPrice.priceUsd,
          tokenPrice.timestamp,
          this.chainId,
          PriceSource.COINGECKO,
          null, // tokenAddress is optional
          {
            marketCap: tokenPrice.marketCap,
            volume24h: tokenPrice.volume24h,
            priceChangePercentage24h: tokenPrice.priceChangePercentage24h
          }
        );
      }
    } catch (error) {
      this.logger.error('Failed to update prices:', error);
      // Don't throw the error to prevent the cron job from crashing
    }
  }

  async getLatestPrice(tokenSymbol: string): Promise<number | null> {
    try {
      const price = await this.priceRepository.getLatestPrice(tokenSymbol, this.chainId);
      return price?.priceUsd ?? null;
    } catch (error) {
      this.logger.error(`Failed to get latest price for ${tokenSymbol}:`, error);
      return null;
    }
  }

  async getPriceHistory(
    tokenSymbol: string,
    startTime: Date,
    endTime: Date
  ): Promise<any[]> {
    try {
      return await this.priceRepository.getPriceHistory(tokenSymbol, startTime, endTime, this.chainId);
    } catch (error) {
      this.logger.error(`Failed to get price history for ${tokenSymbol}:`, error);
      return [];
    }
  }

  async getAggregatedPrices(
    tokenSymbol: string,
    startTime: Date,
    endTime: Date,
    interval: string
  ): Promise<any[]> {
    try {
      return await this.priceRepository.getAggregatedPrices(
        tokenSymbol,
        startTime,
        endTime,
        interval,
        this.chainId
      );
    } catch (error) {
      this.logger.error(`Failed to get aggregated prices for ${tokenSymbol}:`, error);
      return [];
    }
  }
} 