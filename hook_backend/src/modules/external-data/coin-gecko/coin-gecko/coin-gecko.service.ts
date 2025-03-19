import { Injectable, Logger, HttpException, HttpStatus } from '@nestjs/common';
import axios, { AxiosInstance } from 'axios';
import { ConfigService } from '@nestjs/config';

export interface TokenPriceData {
  symbol: string;
  priceUsd: number;
  lastUpdatedAt: Date;
  marketCap?: number;
  volume24h?: number;
  priceChangePercentage24h?: number;
}

@Injectable()
export class CoinGeckoService {
  private readonly logger = new Logger(CoinGeckoService.name);
  private readonly apiClient: AxiosInstance;
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly rateLimitPerMinute: number;
  private requestCount: number = 0;
  private rateLimitResetTime: Date = new Date();
  
  // Token ID mapping (CoinGecko uses IDs, not symbols)
  private readonly tokenIdMap = {
    'ETH': 'ethereum',
    'USDC': 'usd-coin',
    'BTC': 'bitcoin',
    'USDT': 'tether',
    'DAI': 'dai',
    // Add more tokens as needed
  };

  constructor(private configService: ConfigService) {
    this.apiKey = this.configService.get<string>('COINGECKO_API_KEY', '');
    this.baseUrl = 'https://api.coingecko.com/api/v3';
    this.rateLimitPerMinute = this.apiKey ? 500 : 10; // Free tier: 10-50/min, Pro: 500/min
    
    this.apiClient = axios.create({
      baseURL: this.baseUrl,
      timeout: 10000, // 10 seconds
      headers: this.apiKey ? { 'x-cg-pro-api-key': this.apiKey } : {},
    });
    
    // Reset rate limit counter every minute
    setInterval(() => {
      this.requestCount = 0;
      this.rateLimitResetTime = new Date(Date.now() + 60000);
    }, 60000);
  }

  private async checkRateLimit(): Promise<void> {
    if (this.requestCount >= this.rateLimitPerMinute) {
      const waitTime = this.rateLimitResetTime.getTime() - Date.now();
      if (waitTime > 0) {
        this.logger.warn(`Rate limit reached. Waiting ${waitTime}ms before next request.`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
      this.requestCount = 0;
    }
    this.requestCount++;
  }

  private getTokenId(symbol: string): string {
    const tokenId = this.tokenIdMap[symbol.toUpperCase()];
    if (!tokenId) {
      throw new HttpException(`Token symbol ${symbol} not supported`, HttpStatus.BAD_REQUEST);
    }
    return tokenId;
  }

  async getTokenPrice(symbol: string): Promise<TokenPriceData> {
    try {
      await this.checkRateLimit();
      
      const tokenId = this.getTokenId(symbol);
      
      const response = await this.apiClient.get('/simple/price', {
        params: {
          ids: tokenId,
          vs_currencies: 'usd',
          include_market_cap: true,
          include_24hr_vol: true,
          include_24hr_change: true,
          include_last_updated_at: true,
        },
      });
      
      if (!response.data || !response.data[tokenId]) {
        throw new HttpException(`Failed to fetch price for ${symbol}`, HttpStatus.NOT_FOUND);
      }
      
      const data = response.data[tokenId];
      
      return {
        symbol: symbol.toUpperCase(),
        priceUsd: data.usd,
        lastUpdatedAt: new Date(data.last_updated_at * 1000),
        marketCap: data.usd_market_cap,
        volume24h: data.usd_24h_vol,
        priceChangePercentage24h: data.usd_24h_change,
      };
    } catch (error) {
      this.logger.error(`Error fetching price for ${symbol}: ${error.message}`, error.stack);
      throw new HttpException(
        `Failed to fetch price data: ${error.message}`,
        error.response?.status || HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async getMultipleTokenPrices(symbols: string[]): Promise<TokenPriceData[]> {
    try {
      await this.checkRateLimit();
      
      const tokenIds = symbols.map(symbol => this.getTokenId(symbol));
      
      const response = await this.apiClient.get('/simple/price', {
        params: {
          ids: tokenIds.join(','),
          vs_currencies: 'usd',
          include_market_cap: true,
          include_24hr_vol: true,
          include_24hr_change: true,
          include_last_updated_at: true,
        },
      });

      const data = symbols.map((symbol, index) => {
        const tokenId = tokenIds[index];
        const data = response.data[tokenId];

        if (!data) {
          this.logger.warn(`No data returned for ${symbol}`);
          return null;
        }

        return {
          symbol: symbol.toUpperCase(),
          priceUsd: data.usd,
          lastUpdatedAt: new Date(data.last_updated_at * 1000),
          marketCap: data.usd_market_cap,
          volume24h: data.usd_24h_vol,
          priceChangePercentage24h: data.usd_24h_change,
        };
      });
      return data.filter((t) => t !== null);
    } catch (error) {
      this.logger.error(`Error fetching multiple prices: ${error.message}`, error.stack);
      throw new HttpException(
        `Failed to fetch multiple price data: ${error.message}`,
        error.response?.status || HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
