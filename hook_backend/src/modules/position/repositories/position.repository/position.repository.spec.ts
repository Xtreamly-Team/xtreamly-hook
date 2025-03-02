import { PositionRepository } from './position.repository';
import { Test, TestingModule } from '@nestjs/testing';
import { DataSource, Repository, UpdateResult } from 'typeorm';
import { Position, PositionStatus } from '@modules/position/entities/position.entity/position.entity';
import { User } from '@modules/user/entities/user.entity/user.entity';

// Mock User type that matches the entity
const mockUser: User = {
  id: 'user-123',
  walletAddress: '0x123',
  email: '',  // Empty string instead of undefined
  isActive: true,
  positions: [],
  createdAt: new Date(),
  updatedAt: new Date()
};

// Mock Position that matches the entity and service usage
const mockPosition: Position = {
  id: 'position-123',
  userId: 'user-123',
  tokenA: 'ETH',
  tokenB: 'USDC',
  amountA: 1.5,
  amountB: 3000,
  lowerTick: 100,
  upperTick: 200,
  hedgeAmount: 0.75,
  status: PositionStatus.ACTIVE,
  uniswapPositionId: '',  // Empty string instead of undefined
  gmxPositionId: '',      // Empty string instead of undefined
  metadata: {},
  createdAt: new Date(),
  updatedAt: new Date(),
  user: mockUser
};

const mockDataSource = {
  createEntityManager: jest.fn(),
};

const mockUpdateResult: UpdateResult = {
  affected: 1,
  generatedMaps: [],
  raw: {}
};

describe('PositionRepository', () => {
  let repository: PositionRepository;
  let dataSource: DataSource;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        PositionRepository,
        {
          provide: DataSource,
          useValue: mockDataSource,
        },
      ],
    }).compile();

    repository = module.get<PositionRepository>(PositionRepository);
    dataSource = module.get<DataSource>(DataSource);

    // Mock repository methods
    jest.spyOn(repository, 'find').mockResolvedValue([mockPosition]);
    jest.spyOn(repository, 'findOne').mockResolvedValue(mockPosition);
    jest.spyOn(repository, 'create').mockReturnValue(mockPosition);
    jest.spyOn(repository, 'save').mockResolvedValue(mockPosition);
    jest.spyOn(repository, 'update').mockResolvedValue(mockUpdateResult);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('findActivePositionsByUserId', () => {
    it('should return active positions for a user', async () => {
      const userId = 'user-123';
      
      const result = await repository.findActivePositionsByUserId(userId);
      
      expect(repository.find).toHaveBeenCalledWith({
        where: {
          userId,
          status: PositionStatus.ACTIVE,
        },
        relations: ['user'],
      });
      expect(result).toEqual([mockPosition]);
    });
  });

  describe('findPositionById', () => {
    it('should return a position by id', async () => {
      const id = 'position-123';
      
      const result = await repository.findPositionById(id);
      
      expect(repository.findOne).toHaveBeenCalledWith({
        where: { id },
        relations: ['user'],
      });
      expect(result).toEqual(mockPosition);
    });
  });

  describe('createPosition', () => {
    it('should create and save a new position', async () => {
      const positionData: Partial<Position> = {
        userId: 'user-123',
        tokenA: 'ETH',
        tokenB: 'USDC',
        amountA: 1.5,
        amountB: 3000,
      };
      
      const result = await repository.createPosition(positionData);
      
      expect(repository.create).toHaveBeenCalledWith(positionData);
      expect(repository.save).toHaveBeenCalledWith(mockPosition);
      expect(result).toEqual(mockPosition);
    });
  });

  describe('updatePositionStatus', () => {
    it('should update a position status', async () => {
      const id = 'position-123';
      const status = PositionStatus.CLOSED;
      
      await repository.updatePositionStatus(id, status);
      
      expect(repository.update).toHaveBeenCalledWith(id, { status });
    });
  });

  describe('findPositionsForRebalancing', () => {
    it('should return active positions for rebalancing', async () => {
      const result = await repository.findPositionsForRebalancing();
      
      expect(repository.find).toHaveBeenCalledWith({
        where: { status: PositionStatus.ACTIVE },
        relations: ['user'],
      });
      expect(result).toEqual([mockPosition]);
    });
  });
});
