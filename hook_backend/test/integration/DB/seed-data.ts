import { DataSource } from 'typeorm';
import { databaseConfig } from '../../../src/config/database.config';
import { User } from '../../../src/modules/user/entities/user.entity/user.entity';
import { Position, PositionStatus } from '../../../src/modules/position/entities/position.entity/position.entity';
import { PositionHistory } from '../../../src/modules/position/entities/position-history.entity/position-history.entity';

async function seedDatabase() {
  const dataSource = new DataSource({
    ...databaseConfig,
    logging: true,
  } as any);

  try {
    await dataSource.initialize();
    console.log('✅ Database connection successful');

    // Create test users
    const users = [
      {
        walletAddress: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
        email: 'user1@example.com',
        isActive: true,
      },
      {
        walletAddress: '0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199',
        email: 'user2@example.com',
        isActive: true,
      },
      {
        walletAddress: '0xdD2FD4581271e230360230F9337D5c0430Bf44C0',
        email: '',
        isActive: true,
      },
    ];

    console.log('Creating test users...');
    const createdUsers: User[] = [];
    
    for (const userData of users) {
      const existingUser = await dataSource.getRepository(User).findOne({
        where: { walletAddress: userData.walletAddress },
      });
      
      if (!existingUser) {
        const user = dataSource.getRepository(User).create({
          walletAddress: userData.walletAddress,
          email: userData.email,
          isActive: userData.isActive
        });
        const savedUser = await dataSource.getRepository(User).save(user);
        createdUsers.push(savedUser);
        console.log(`Created user: ${savedUser.walletAddress}`);
      } else {
        createdUsers.push(existingUser);
        console.log(`User already exists: ${existingUser.walletAddress}`);
      }
    }

    // Create test positions
    const positions = [
      {
        userId: createdUsers[0].id,
        tokenA: 'ETH',
        tokenB: 'USDC',
        amountA: 1.5,
        amountB: 3000,
        lowerTick: 100,
        upperTick: 200,
        hedgeAmount: 0.75,
        status: PositionStatus.ACTIVE,
        metadata: { source: 'test-data' },
      },
      {
        userId: createdUsers[0].id,
        tokenA: 'BTC',
        tokenB: 'USDT',
        amountA: 0.1,
        amountB: 4000,
        lowerTick: 150,
        upperTick: 250,
        hedgeAmount: 0.05,
        status: PositionStatus.ACTIVE,
        metadata: { source: 'test-data' },
      },
      {
        userId: createdUsers[1].id,
        tokenA: 'ETH',
        tokenB: 'DAI',
        amountA: 2.0,
        amountB: 4000,
        lowerTick: 120,
        upperTick: 220,
        hedgeAmount: 1.0,
        status: PositionStatus.PENDING,
        metadata: { source: 'test-data' },
      },
    ];

    console.log('Creating test positions...');
    const createdPositions: Position[] = [];
    
    for (const positionData of positions) {
      const position = dataSource.getRepository(Position).create({
        userId: positionData.userId,
        tokenA: positionData.tokenA,
        tokenB: positionData.tokenB,
        amountA: positionData.amountA,
        amountB: positionData.amountB,
        lowerTick: positionData.lowerTick,
        upperTick: positionData.upperTick,
        hedgeAmount: positionData.hedgeAmount,
        status: positionData.status,
        metadata: positionData.metadata,
        uniswapPositionId: '',
        gmxPositionId: ''
      });
      const savedPosition = await dataSource.getRepository(Position).save(position);
      createdPositions.push(savedPosition);
      console.log(`Created position: ${savedPosition.id}`);
    }

    // Create position history entries
    console.log('Creating position history entries...');
    
    // Create entries for the past 30 days
    const now = new Date();
    const startDate = new Date(now);
    startDate.setDate(now.getDate() - 30);
    
    for (const position of createdPositions) {
      // Skip pending positions
      if (position.status !== PositionStatus.ACTIVE) continue;
      
      // Create daily entries
      for (let i = 0; i <= 30; i++) {
        const date = new Date(startDate);
        date.setDate(startDate.getDate() + i);
        
        // Simulate some price movement
        const priceMultiplier = 1 + (Math.sin(i / 5) * 0.1); // +/- 10% sinusoidal movement
        
        const entry = dataSource.getRepository(PositionHistory).create({
          positionId: position.id,
          tokenAValue: Number(position.amountA) * priceMultiplier,
          tokenBValue: Number(position.amountB),
          hedgeValue: Number(position.hedgeAmount || 0) * priceMultiplier,
          netValue: (Number(position.amountA) * priceMultiplier) + Number(position.amountB),
          timestamp: date,
          metadata: { day: i, source: 'test-data' }
        });
        
        await dataSource.getRepository(PositionHistory).save(entry);
      }
      
      console.log(`Created 31 history entries for position: ${position.id}`);
    }

    console.log('✅ Database seeding completed successfully');
  } catch (error) {
    console.error('❌ Database seeding failed:', error);
  } finally {
    if (dataSource.isInitialized) {
      await dataSource.destroy();
    }
  }
}

seedDatabase().catch(console.error); 