import { Controller, Get } from '@nestjs/common';
import { PositionService } from '@modules/position/services/position/position.service';
import { AuthUser, UserContext } from '@modules/auth/guards/auth.guard';
import { PositionResponseDto } from '@modules/position/dto/position.dto/position.dto';
import {GMX} from "@app/utils/GMX";
import {shortIt, closeShort} from "@app/utils/gmx_create_order";

@Controller('position')
export class PositionController {
  constructor(private readonly positionService: PositionService) {}

  @Get('/positions')
  async getPositions(
    @UserContext() user: AuthUser,
  ): Promise<PositionResponseDto[]> {
    return this.positionService.getUserPositions(user.user.id);
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
  // async testGMXCreateOrder(@UserContext() user: AuthUser): Promise<any> {
  async testGMXCreateOrder(): Promise<any> {
    // const privateKey = user.firebaseUser.private;
    await shortIt();
  }

  @Get('/gmx-close-short')
  async testGMXCloseShort(): Promise<any> {
    await closeShort();
  }
}
