import { Controller, Get } from '@nestjs/common';
import { PositionService } from '@modules/position/services/position/position.service';
import { AuthUser, UserContext } from '@modules/auth/guards/auth.guard';
import { PositionResponseDto } from '@modules/position/dto/position.dto/position.dto';
import {GMX} from "@app/utils/GMX";

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
}
