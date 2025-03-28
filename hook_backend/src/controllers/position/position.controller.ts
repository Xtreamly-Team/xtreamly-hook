import { Controller, Get, Post, Body, UseGuards, NotFoundException, Param } from '@nestjs/common';
import { PositionService } from '@modules/position/services/position/position.service';
import { UserService } from '@modules/user/services/user/user.service';
import { AuthUser, UserContext } from '@modules/auth/guards/auth.guard';
import { PositionResponseDto } from '@modules/position/dto/position.dto/position.dto';
import { CreatePositionDto } from '@modules/position/dto/position.dto/create-position.dto';
import { GMX } from "@app/utils/GMX";
import { shortIt, closeShort } from "@app/modules/gmx-hedgooor/gmx-hedgooor/gmx_create_order.service";

@Controller('position')
export class PositionController {
  constructor(
    private readonly positionService: PositionService,
    private readonly userService: UserService
  ) {}

  @Get('/positions')
  async getPositions(
    @UserContext() user: AuthUser,
  ): Promise<PositionResponseDto[]> {
    return this.positionService.getUserPositions(user.user.id);
  }

  @Post('/create-LP-position')
  async createPosition(
    @Body() createPositionDto: CreatePositionDto
  ): Promise<PositionResponseDto> {
    const user = await this.userService.findByWalletAddress(createPositionDto.walletAddress);

    const position = await this.positionService.createPosition({
      userId: user.id,
      tokenA: createPositionDto.tokenA,
      tokenB: createPositionDto.tokenB,
      amountA: createPositionDto.amountA,
      amountB: createPositionDto.amountB,
      lowerTick: createPositionDto.lowerTick,
      upperTick: createPositionDto.upperTick,
      hedgeAmount: createPositionDto.hedgeAmount,
      status: createPositionDto.status,
      uniswapPositionId: createPositionDto.uniswapPositionId,
      gmxPositionId: createPositionDto.gmxPositionId,
      metadata: createPositionDto.metadata,
    });
    
    return {
      id: position.id,
      userId: position.userId,
      tokenA: position.tokenA,
      tokenB: position.tokenB,
      amountA: Number(position.amountA),
      amountB: Number(position.amountB),
      lowerTick: Number(position.lowerTick),
      upperTick: Number(position.upperTick),
      hedgeAmount: Number(position.hedgeAmount),
      status: position.status,
      uniswapPositionId: position.uniswapPositionId,
      gmxPositionId: position.gmxPositionId,
      createdAt: position.createdAt,
      updatedAt: position.updatedAt
    };
  }

  @Get('/gmx')
  async testGMX(@UserContext() user: AuthUser): Promise<any> {
    const trades = await new GMX(user.firebaseUser.private).getTrades();
    return trades.map((t) => t.transaction.hash);
  }

  // @Get('/gmx-positions')
  // async testGMXPositions(@UserContext() user: AuthUser): Promise<any> {
  //   const positions = await new GMX(user.firebaseUser.private).getPositions();
  //   return positions;
  // }

  @Get('/gmx-create-order') 
  async testGMXCreateOrder(): Promise<any> {
    // const privateKey = user.firebaseUser.private;
    await shortIt();
  }

  @Get('/gmx-close-short')
  async testGMXCloseShort(): Promise<any> {
    await closeShort();
  }

  @Get('/wallet/:walletAddress')
  async getPositionsByWalletAddress(
    @Param('walletAddress') walletAddress: string
  ): Promise<PositionResponseDto[]> {
    const user = await this.userService.findByWalletAddress(walletAddress);
    const positions = await this.positionService.getUserPositions(user.id);
    
    return positions;
  }
}
