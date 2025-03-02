import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { User } from './entities/user.entity/user.entity';
import { UserRepository } from './repositories/user.repository/user.repository';
import { UserService } from './services/user/user.service';
import { UserController } from '../../controllers/user/user.controller';

@Module({
  imports: [TypeOrmModule.forFeature([User])],
  providers: [UserRepository, UserService],
  controllers: [UserController],
  exports: [UserRepository, UserService],
})
export class UserModule {}