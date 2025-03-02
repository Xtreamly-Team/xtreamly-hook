import { Injectable } from '@nestjs/common';
import { DataSource, Repository, FindOptionsWhere } from 'typeorm';
import { Position, PositionStatus } from '../../entities/position.entity/position.entity';

@Injectable()
export class PositionRepository extends Repository<Position> {
  constructor(private dataSource: DataSource) {
    super(Position, dataSource.createEntityManager());
  }

  async findActivePositionsByUserId(userId: string): Promise<Position[]> {
    return this.find({
      where: {
        userId,
        status: PositionStatus.ACTIVE,
      },
      relations: ['user'],
    });
  }

  async findPositionById(id: string): Promise<Position | null> {
    return this.findOne({
      where: { id },
      relations: ['user'],
    });
  }

  async createPosition(positionData: Partial<Position>): Promise<Position> {
    const position = this.create(positionData);
    return this.save(position);
  }

  async updatePositionStatus(id: string, status: PositionStatus): Promise<void> {
    await this.update(id, { status });
  }

  async findPositionsForRebalancing(): Promise<Position[]> {
    return this.find({
      where: { status: PositionStatus.ACTIVE },
      relations: ['user'],
    });
  }
}