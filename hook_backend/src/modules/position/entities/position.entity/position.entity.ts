import { Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn, CreateDateColumn, UpdateDateColumn, Index } from 'typeorm';
import { User } from '@modules/user/entities/user.entity/user.entity';

export enum PositionStatus {
  PENDING = 'pending',
  ACTIVE = 'active',
  REBALANCING = 'rebalancing',
  CLOSED = 'closed',
}

@Entity('positions')
export class Position {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  userId: string;

  @ManyToOne(() => User, (user) => user.positions)
  @JoinColumn({ name: 'userId' })
  user: User;

  @Column()
  tokenA: string;

  @Column()
  tokenB: string;

  @Column({ type: 'decimal', precision: 24, scale: 8 })
  amountA: number;

  @Column({ type: 'decimal', precision: 24, scale: 8 })
  amountB: number;

  @Column({ type: 'decimal', precision: 24, scale: 8 })
  lowerTick: number;

  @Column({ type: 'decimal', precision: 24, scale: 8 })
  upperTick: number;

  @Column({ type: 'decimal', precision: 24, scale: 8, nullable: true })
  hedgeAmount: number;

  @Column({ type: 'varchar', default: PositionStatus.PENDING })
  status: PositionStatus;

  @Column({ nullable: true })
  uniswapPositionId: string;

  @Column({ nullable: true })
  gmxPositionId: string;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @CreateDateColumn()
  @Index()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}