import { Test } from '@nestjs/testing';
import { DataSource } from 'typeorm';
import { UserRepository } from './user.repository';

describe('UserRepository', () => {
  let repository: UserRepository;
  
  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [
        UserRepository,
        {
          provide: DataSource,
          useValue: {
            createEntityManager: jest.fn()
          }
        }
      ],
    }).compile();

    repository = module.get<UserRepository>(UserRepository);
  });

  it('should be defined', () => {
    expect(repository).toBeDefined();
  });
});
