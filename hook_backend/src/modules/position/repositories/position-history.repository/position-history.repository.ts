import { Injectable } from '@nestjs/common';
import { DataSource, Repository, Between } from 'typeorm';
import { PositionHistory } from '../../entities/position-history.entity/position-history.entity';

@Injectable()
export class PositionHistoryRepository extends Repository<PositionHistory> {
  constructor(private dataSource: DataSource) {
    super(PositionHistory, dataSource.createEntityManager());
  }

  async createHistoryEntry(historyData: Partial<PositionHistory>): Promise<PositionHistory> {
    const historyEntry = this.create(historyData);
    return this.save(historyEntry);
  }

  async getPositionHistoryByTimeRange(
    positionId: string,
    startDate: Date,
    endDate: Date,
  ): Promise<PositionHistory[]> {
    return this.find({
      where: {
        positionId,
        timestamp: Between(startDate, endDate),
      },
      order: {
        timestamp: 'ASC',
      },
    });
  }

  async getLatestPositionHistory(positionId: string): Promise<PositionHistory | null> {
    return this.findOne({
      where: { positionId },
      order: { timestamp: 'DESC' },
    });
  }

  /**
   * Uses TimescaleDB's time_bucket function to aggregate position history data
   */
  async getAggregatedPositionHistory(positionId: string, interval: string): Promise<any[]> {
    return this.query(`
      SELECT 
        time_bucket('${interval}', timestamp) as time_bucket,
        AVG(token_a_value) as avg_token_a_value,
        AVG(token_b_value) as avg_token_b_value,
        AVG(hedge_value) as avg_hedge_value,
        AVG(net_value) as avg_net_value,
        MIN(net_value) as min_net_value,
        MAX(net_value) as max_net_value
      FROM position_history
      WHERE position_id = $1
      GROUP BY time_bucket
      ORDER BY time_bucket ASC
    `, [positionId]);
  }
}