import { PositionHistoryRepository } from './position-history.repository';
import { Test } from '@nestjs/testing';
import { DataSource, Repository, Between } from 'typeorm';
import { PositionHistory } from '../../entities/position-history.entity/position-history.entity';

describe('PositionHistoryRepository', () => {
  let repository: PositionHistoryRepository;
  let dataSource: DataSource;
  
  const mockPositionHistory = {
    id: '123',
    positionId: 'position-123',
    tokenAValue: 100,
    tokenBValue: 200,
    hedgeValue: 50,
    netValue: 350,
    timestamp: new Date(),
    metadata: { test: 'data' }
  };
  
  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [
        PositionHistoryRepository,
        {
          provide: DataSource,
          useValue: {
            createEntityManager: jest.fn(),
            query: jest.fn()
          }
        }
      ],
    }).compile();

    repository = module.get<PositionHistoryRepository>(PositionHistoryRepository);
    dataSource = module.get<DataSource>(DataSource);
    
    // Mock repository methods
    jest.spyOn(repository, 'create').mockReturnValue(mockPositionHistory as any);
    jest.spyOn(repository, 'save').mockResolvedValue(mockPositionHistory as any);
    jest.spyOn(repository, 'find').mockResolvedValue([mockPositionHistory] as any);
    jest.spyOn(repository, 'findOne').mockResolvedValue(mockPositionHistory as any);
    jest.spyOn(dataSource, 'query').mockResolvedValue([
      {
        time_bucket: '2023-01-01T00:00:00.000Z',
        avg_token_a_value: 100,
        avg_token_b_value: 200,
        avg_hedge_value: 50,
        avg_net_value: 350,
        min_net_value: 300,
        max_net_value: 400
      }
    ]);
  });

  it('should be defined', () => {
    expect(repository).toBeDefined();
  });
  
  describe('createHistoryEntry', () => {
    it('should create and save a history entry', async () => {
      const historyData = {
        positionId: 'position-123',
        tokenAValue: 100,
        tokenBValue: 200,
        hedgeValue: 50,
        netValue: 350
      };
      
      const result = await repository.createHistoryEntry(historyData);
      
      expect(repository.create).toHaveBeenCalledWith(historyData);
      expect(repository.save).toHaveBeenCalledWith(mockPositionHistory);
      expect(result).toEqual(mockPositionHistory);
    });
  });
  
  describe('getPositionHistoryByTimeRange', () => {
    it('should return position history within a time range', async () => {
      const positionId = 'position-123';
      const startDate = new Date('2023-01-01');
      const endDate = new Date('2023-01-31');
      
      const result = await repository.getPositionHistoryByTimeRange(positionId, startDate, endDate);
      
      expect(repository.find).toHaveBeenCalledWith({
        where: {
          positionId,
          timestamp: Between(startDate, endDate),
        },
        order: {
          timestamp: 'ASC',
        },
      });
      expect(result).toEqual([mockPositionHistory]);
    });
  });
  
  describe('getLatestPositionHistory', () => {
    it('should return the latest position history entry', async () => {
      const positionId = 'position-123';
      
      const result = await repository.getLatestPositionHistory(positionId);
      
      expect(repository.findOne).toHaveBeenCalledWith({
        where: { positionId },
        order: { timestamp: 'DESC' },
      });
      expect(result).toEqual(mockPositionHistory);
    });
  });
  
  describe('getAggregatedPositionHistory', () => {
    it('should return aggregated position history data', async () => {
      const positionId = 'position-123';
      const interval = '1 day';
      
      const result = await repository.getAggregatedPositionHistory(positionId, interval);
      
      expect(dataSource.query).toHaveBeenCalledWith(
        expect.stringContaining(`time_bucket('${interval}', timestamp) as time_bucket`),
        [positionId]
      );
      expect(result).toEqual([
        {
          time_bucket: '2023-01-01T00:00:00.000Z',
          avg_token_a_value: 100,
          avg_token_b_value: 200,
          avg_hedge_value: 50,
          avg_net_value: 350,
          min_net_value: 300,
          max_net_value: 400
        }
      ]);
    });
  });
});


