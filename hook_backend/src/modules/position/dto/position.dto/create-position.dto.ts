import { IsNotEmpty, IsNumber, IsString, IsOptional, IsEnum, IsEthereumAddress } from 'class-validator';
import { PositionStatus } from '@modules/position/entities/position.entity/position.entity';

export class CreatePositionDto {
  @IsEthereumAddress()
  @IsNotEmpty()
  walletAddress: string;

  @IsString()
  @IsNotEmpty()
  tokenA: string;

  @IsString()
  @IsNotEmpty()
  tokenB: string;

  @IsNumber()
  @IsNotEmpty()
  amountA: number;

  @IsNumber()
  @IsNotEmpty()
  amountB: number;

  @IsNumber()
  @IsNotEmpty()
  lowerTick: number;

  @IsNumber()
  @IsNotEmpty()
  upperTick: number;

  @IsNumber()
  @IsNotEmpty()
  hedgeAmount: number;

  @IsEnum(PositionStatus)
  @IsOptional()
  status?: PositionStatus;

  @IsString()
  @IsOptional()
  uniswapPositionId?: string;

  @IsString()
  @IsOptional()
  gmxPositionId?: string;

  @IsOptional()
  metadata?: Record<string, any>;
} 