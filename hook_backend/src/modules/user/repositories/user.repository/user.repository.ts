import { Injectable } from '@nestjs/common';
import { DataSource, Repository } from 'typeorm';
import { User } from '@modules/user/entities/user.entity/user.entity';

@Injectable()
export class UserRepository extends Repository<User> {
  constructor(private dataSource: DataSource) {
    super(User, dataSource.createEntityManager());
  }

  async findByWalletAddress(walletAddress: string): Promise<User | null> {
    return this.findOne({ where: { walletAddress } });
  }

  async createUser(walletAddress: string, email?: string): Promise<User> {
    const user = this.create({
      walletAddress,
      email,
      isActive: true,
    });
    
    return this.save(user);
  }
}