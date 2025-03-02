import { Injectable, NotFoundException } from '@nestjs/common';
import { PositionRepository } from '../../repositories/position.repository/position.repository';
import { PositionHistoryRepository } from '../../repositories/position-history.repository/position-history.repository';
import { Position, PositionStatus } from '../../entities/position.entity/position.entity';
import { PositionHistory } from '../../entities/position-history.entity/position-history.entity';
import { UserService } from '../../../user/services/user/user.service';

@Injectable()
export class PositionService {
  constructor(
    private positionRepository: PositionRepository,
    private positionHistoryRepository: PositionHistoryRepository,
    private userService: UserService,
  ) {}

  async getPositionById(id: string): Promise<Position> {
    const position = await this.positionRepository.findPositionById(id);
    if (!position) {
      throw new NotFoundException(`Position with ID ${id} not found`);
    }
    return position;
  }

  async getUserPositions(userId: string): Promise<Position[]> {
    await this.userService.getUserById(userId); // Validate user exists
    return this.positionRepository.findActivePositionsByUserId(userId);
  }

  async createPosition(positionData: Partial<Position>): Promise<Position> {
    const position = await this.positionRepository.createPosition(positionData);
    
    // Create initial history entry
    await this.recordPositionHistory(position.id, {
      tokenAValue: Number(position.amountA),
      tokenBValue: Number(position.amountB),
      hedgeValue: position.hedgeAmount ? Number(position.hedgeAmount) : 0,
      netValue: Number(position.amountA) + Number(position.amountB),
    });
    
    return position;
  }

  async updatePositionStatus(id: string, status: PositionStatus): Promise<void> {
    const position = await this.getPositionById(id);
    await this.positionRepository.updatePositionStatus(id, status);
  }

  async recordPositionHistory(
    positionId: string,
    data: {
      tokenAValue: number;
      tokenBValue: number;
      hedgeValue: number;
      netValue: number;
      metadata?: Record<string, any>;
    },
  ): Promise<PositionHistory> {
    return this.positionHistoryRepository.createHistoryEntry({
      positionId,
      tokenAValue: data.tokenAValue,
      tokenBValue: data.tokenBValue,
      hedgeValue: data.hedgeValue,
      netValue: data.netValue,
      metadata: data.metadata,
    });
  }

  async getPositionHistory(
    positionId: string,
    startDate: Date,
    endDate: Date,
  ): Promise<PositionHistory[]> {
    await this.getPositionById(positionId); // Validate position exists
    return this.positionHistoryRepository.getPositionHistoryByTimeRange(
      positionId,
      startDate,
      endDate,
    );
  }

  async getAggregatedPositionHistory(positionId: string, interval: string): Promise<any[]> {
    await this.getPositionById(positionId); // Validate position exists
    return this.positionHistoryRepository.getAggregatedPositionHistory(positionId, interval);
  }

  async getPositionsForRebalancing(): Promise<Position[]> {
    return this.positionRepository.findPositionsForRebalancing();
  }
}