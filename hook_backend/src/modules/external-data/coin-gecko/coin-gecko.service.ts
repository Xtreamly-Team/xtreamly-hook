import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios, { AxiosInstance } from 'axios';

export interface TokenPriceData {
  symbol: string;
  priceUsd: number;
  timestamp: Date;
  marketCap?: number;
  volume24h?: number;
  priceChangePercentage24h?: number;
}

@Injectable()
export class CoinGeckoService {
  private readonly logger = new Logger(CoinGeckoService.name);
  private readonly axiosInstance: AxiosInstance;
  private readonly apiKey: string | null;
  private readonly baseUrl: string;
  private readonly requestsPerMinute: number;
  private requestCount = 0;
  private lastResetTime = Date.now();

  private readonly tokenIdMap: Record<string, string> = {
    'ETH': 'ethereum',
    'USDC': 'usd-coin',
    // Add more mappings as needed
  };

  constructor(private readonly configService: ConfigService) {
    this.apiKey = this.configService.get<string>('COINGECKO_API_KEY') || null;
    this.baseUrl = 'https://api.coingecko.com/api/v3';
    this.requestsPerMinute = this.apiKey ? 450 : 10;

    this.axiosInstance = axios.create({
      baseURL: this.baseUrl,
      headers: this.apiKey ? { 'x-cg-pro-api-key': this.apiKey } : {},
    });

    // Reset rate limit counter every minute
    setInterval(() => {
      this.requestCount = 0;
      this.lastResetTime = Date.now();
    }, 60000);
  }

  private async checkRateLimit() {
    if (this.requestCount >= this.requestsPerMinute) {
      const timeToWait = 60000 - (Date.now() - this.lastResetTime);
      if (timeToWait > 0) {
        await new Promise(resolve => setTimeout(resolve, timeToWait));
      }
      this.requestCount = 0;
      this.lastResetTime = Date.now();
    }
    this.requestCount++;
  }

  private getTokenId(symbol: string): string {
    const tokenId = this.tokenIdMap[symbol.toUpperCase()];
    if (!tokenId) {
      throw new Error(`Unsupported token symbol: ${symbol}`);
    }
    return tokenId;
  }

  async getTokenPrice(symbol: string): Promise<TokenPriceData> {
    try {
      await this.checkRateLimit();
      const tokenId = this.getTokenId(symbol);
      
      const response = await this.axiosInstance.get(`/simple/price`, {
        params: {
          ids: tokenId,
          vs_currencies: 'usd',
          include_market_cap: true,
          include_24hr_vol: true,
          include_24hr_change: true,
          include_last_updated_at: true
        }
      });

      const data = response.data[tokenId];
      if (!data) {
        throw new Error(`No data returned for token ${symbol}`);
      }

      return {
        symbol,
        priceUsd: data.usd,
        timestamp: new Date(data.last_updated_at * 1000),
        marketCap: data.usd_market_cap,
        volume24h: data.usd_24h_vol,
        priceChangePercentage24h: data.usd_24h_change
      };
    } catch (error) {
      this.logger.error(`Error fetching price for ${symbol}: ${error.message}`);
      throw error;
    }
  }

  async getMultipleTokenPrices(symbols: string[]): Promise<TokenPriceData[]> {
    try {
      await this.checkRateLimit();
      const tokenIds = symbols.map(symbol => this.getTokenId(symbol));
      
      const response = await this.axiosInstance.get(`/simple/price`, {
        params: {
          ids: tokenIds.join(','),
          vs_currencies: 'usd',
          include_market_cap: true,
          include_24hr_vol: true,
          include_24hr_change: true,
          include_last_updated_at: true
        }
      });

      return symbols.map((symbol): TokenPriceData => {
        const tokenId = this.getTokenId(symbol);
        const data = response.data[tokenId];
        if (!data) {
          throw new Error(`No data returned for token ${symbol}`);
        }

        return {
          symbol,
          priceUsd: data.usd,
          timestamp: new Date(data.last_updated_at * 1000),
          marketCap: data.usd_market_cap,
          volume24h: data.usd_24h_vol,
          priceChangePercentage24h: data.usd_24h_change
        };
      });
    } catch (error) {
      this.logger.error(`Error fetching prices for multiple tokens: ${error.message}`);
      throw error;
    }
  }
} 