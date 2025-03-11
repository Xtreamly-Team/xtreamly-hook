import { Entity, Column, PrimaryGeneratedColumn, Index } from 'typeorm';

export enum PriceSource {
  COINGECKO = 'coingecko',
  FALLBACK = 'fallback',
  MANUAL = 'manual'
}

@Entity('token_prices')
export class TokenPrice {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  @Index()
  tokenSymbol: string;

  @Column({ type: 'varchar', length: 42, nullable: true })
  tokenAddress: string | null;

  @Column('decimal', { precision: 24, scale: 8 })
  priceUsd: number;

  @Column({ type: 'timestamp with time zone' })
  @Index()
  timestamp: Date;

  @Column({
    type: 'enum',
    enum: PriceSource,
    default: PriceSource.COINGECKO
  })
  source: PriceSource;

  @Column()
  @Index()
  chainId: number;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any> | null;

  @Column({ type: 'timestamp with time zone', default: () => 'CURRENT_TIMESTAMP' })
  createdAt: Date;
} 