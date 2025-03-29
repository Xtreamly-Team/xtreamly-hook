import { Controller, Get } from '@nestjs/common';
import { PositionService } from '@modules/position/services/position/position.service';
import { UserService } from '@modules/user/services/user/user.service';
import { AuthUser, UserContext } from '@modules/auth/guards/auth.guard';
import { PositionResponseDto } from '@modules/position/dto/position.dto/position.dto';
import { GMX } from "@app/utils/GMX";

@Controller('position')
export class PositionController {
  constructor(
    private readonly positionService: PositionService,
    private readonly userService: UserService,
  ) {}

  @Get('/positions')
  async getPositions(
    @UserContext() user: AuthUser,
  ): Promise<PositionResponseDto[]> {
    return this.positionService.getUserPositions(user.user.id);
  }

  // @Get('/gmx')
  // async testGMX(@UserContext() user: AuthUser): Promise<any> {
  //   const gmx = new GMX(user.firebaseUser.private);
  //   await gmx.init();
  //   const res = await gmx.closeShort();
  //   console.log(res);
  //   return res;
  // }
}
