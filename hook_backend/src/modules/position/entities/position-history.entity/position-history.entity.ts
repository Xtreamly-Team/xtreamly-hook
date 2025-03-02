import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, Index } from 'typeorm';

@Entity('position_history')
export class PositionHistory {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  @Index()
  positionId: string;

  @Column({ type: 'decimal', precision: 24, scale: 8 })
  tokenAValue: number;

  @Column({ type: 'decimal', precision: 24, scale: 8 })
  tokenBValue: number;

  @Column({ type: 'decimal', precision: 24, scale: 8, nullable: true })
  hedgeValue: number;

  @Column({ type: 'decimal', precision: 24, scale: 8, nullable: true })
  netValue: number;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @CreateDateColumn()
  @Index()
  timestamp: Date;
}