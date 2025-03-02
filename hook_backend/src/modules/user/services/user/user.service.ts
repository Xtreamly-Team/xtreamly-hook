import { Injectable, NotFoundException } from '@nestjs/common';
import { UserRepository } from '../../repositories/user.repository/user.repository';
import { User } from '../../entities/user.entity/user.entity';

@Injectable()
export class UserService {
  constructor(private userRepository: UserRepository) {}

  async findByWalletAddress(walletAddress: string): Promise<User> {
    const user = await this.userRepository.findByWalletAddress(walletAddress);
    if (!user) {
      throw new NotFoundException(`User with wallet address ${walletAddress} not found`);
    }
    return user;
  }

  async registerUser(walletAddress: string, email?: string): Promise<User> {
    const existingUser = await this.userRepository.findByWalletAddress(walletAddress);
    if (existingUser) {
      return existingUser;
    }
    
    return this.userRepository.createUser(walletAddress, email);
  }

  async getUserById(id: string): Promise<User> {
    const user = await this.userRepository.findOne({ where: { id } });
    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    return user;
  }
}