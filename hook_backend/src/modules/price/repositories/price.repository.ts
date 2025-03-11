import { Injectable } from '@nestjs/common';
import { DataSource, Repository } from 'typeorm';
import { TokenPrice, PriceSource } from '../entities/token-price.entity';

@Injectable()
export class PriceRepository extends Repository<TokenPrice> {
  constructor(private dataSource: DataSource) {
    super(TokenPrice, dataSource.createEntityManager());
  }

  async savePrice(
    tokenSymbol: string,
    priceUsd: number,
    timestamp: Date,
    chainId: number,
    source: PriceSource = PriceSource.COINGECKO,
    tokenAddress?: string | null,
    metadata?: Record<string, any> | null
  ): Promise<TokenPrice> {
    const price = this.create({
      tokenSymbol,
      priceUsd,
      timestamp,
      chainId,
      source,
      tokenAddress: tokenAddress || null,
      metadata: metadata || null
    });

    return this.save(price);
  }

  async getLatestPrice(tokenSymbol: string, chainId?: number): Promise<TokenPrice | null> {
    const query = this.createQueryBuilder('price')
      .where('price.tokenSymbol = :tokenSymbol', { tokenSymbol })
      .orderBy('price.timestamp', 'DESC')
      .limit(1);

    if (chainId) {
      query.andWhere('price.chainId = :chainId', { chainId });
    }

    return query.getOne();
  }

  async getPriceHistory(
    tokenSymbol: string,
    startTime: Date,
    endTime: Date,
    chainId?: number
  ): Promise<TokenPrice[]> {
    const query = this.createQueryBuilder('price')
      .where('price.tokenSymbol = :tokenSymbol', { tokenSymbol })
      .andWhere('price.timestamp BETWEEN :startTime AND :endTime', { startTime, endTime })
      .orderBy('price.timestamp', 'ASC');

    if (chainId) {
      query.andWhere('price.chainId = :chainId', { chainId });
    }

    return query.getMany();
  }

  async getAggregatedPrices(
    tokenSymbol: string,
    startTime: Date,
    endTime: Date,
    interval: string,
    chainId?: number
  ): Promise<any[]> {
    const query = this.createQueryBuilder('price')
      .select([
        'time_bucket(:interval, price.timestamp) as bucket',
        'AVG(price.priceUsd)::numeric(24,8) as avg_price',
        'MIN(price.priceUsd)::numeric(24,8) as min_price',
        'MAX(price.priceUsd)::numeric(24,8) as max_price',
        'FIRST(price.priceUsd, price.timestamp)::numeric(24,8) as open_price',
        'LAST(price.priceUsd, price.timestamp)::numeric(24,8) as close_price',
        'COUNT(*) as sample_count'
      ])
      .where('price.tokenSymbol = :tokenSymbol', { tokenSymbol })
      .andWhere('price.timestamp BETWEEN :startTime AND :endTime', { startTime, endTime })
      .setParameter('interval', interval)
      .groupBy('bucket')
      .orderBy('bucket', 'ASC');

    if (chainId) {
      query.andWhere('price.chainId = :chainId', { chainId });
    }

    return query.getRawMany();
  }
} 