import { 
  IsString, 
  IsNumber, 
  IsUUID, 
  IsEnum, 
  IsOptional, 
  IsDateString, 
  IsObject, 
  Min 
} from 'class-validator';
import { PositionStatus } from '../../entities/position.entity/position.entity';

export class OpenPositionQuoteDto {
  @IsUUID()
  userId: string;

  @IsString()
  tokenA: string;

  @IsString()
  tokenB: string;

  @IsNumber({ maxDecimalPlaces: 8 })
  @Min(0)
  amountA: number;

  @IsNumber({ maxDecimalPlaces: 8 })
  @Min(0)
  amountB: number;
}

export class PositionResponseDto {
  id: string;
  userId: string;
  tokenA: string;
  tokenB: string;
  amountA: number;
  amountB: number;
  lowerTick: number;
  upperTick: number;
  hedgeAmount: number;
  status: PositionStatus;
  uniswapPositionId?: string;
  gmxPositionId?: string;
  createdAt: Date;
  updatedAt: Date;
}

export class PositionHistoryDto {
  @IsUUID()
  positionId: string;

  @IsDateString()
  startDate: string;

  @IsDateString()
  endDate: string;

  @IsString()
  @IsOptional()
  interval?: string;
}

export class UpdatePositionStatusDto {
  @IsEnum(PositionStatus)
  status: PositionStatus;
}

export class RecordPositionHistoryDto {
  @IsNumber({ maxDecimalPlaces: 8 })
  tokenAValue: number;

  @IsNumber({ maxDecimalPlaces: 8 })
  tokenBValue: number;

  @IsNumber({ maxDecimalPlaces: 8 })
  hedgeValue: number;

  @IsNumber({ maxDecimalPlaces: 8 })
  netValue: number;

  @IsObject()
  @IsOptional()
  metadata?: Record<string, any>;
}