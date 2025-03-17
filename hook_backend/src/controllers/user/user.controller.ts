import { Controller, Post, Body, Get, Param } from '@nestjs/common';
import { UserService } from '@modules/user/services/user/user.service';
import { RegisterUserDto, UserResponseDto } from '@modules/user/dto/user.dto/user.dto';
import {User} from "@modules/auth/guards/auth.guard";
  
@Controller('users')
export class UserController {
  constructor(private readonly userService: UserService) {}

  @Post('register')
  async registerNewUser(@Body() registerUserDto: RegisterUserDto): Promise<UserResponseDto> {
    const user = await this.userService.registerUser(
      registerUserDto.walletAddress,
      registerUserDto.email,
    );
    
    return {
      id: user.id,
      walletAddress: user.walletAddress,
      email: user.email,
      isActive: user.isActive,
      createdAt: user.createdAt,
    };
  }

  @Get(':walletAddress')
  async getUserByWalletAddress(@Param('walletAddress') walletAddress: string): Promise<UserResponseDto> {
    const user = await this.userService.findByWalletAddress(walletAddress);
    
    return {
      id: user.id,
      walletAddress: user.walletAddress,
      email: user.email,
      isActive: user.isActive,
      createdAt: user.createdAt,
    };
  }

  @Get('/')
  async getUser(@User() user: any): Promise<UserResponseDto> {
    return user;
  }
}