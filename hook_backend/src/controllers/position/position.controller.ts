import {Controller, Get} from '@nestjs/common';
import {PositionService} from "@modules/position/services/position/position.service";
import {AuthUser, UserContext} from "@modules/auth/guards/auth.guard";
import {PositionResponseDto} from "@modules/position/dto/position.dto/position.dto";

@Controller('position')
export class PositionController {
    constructor(private readonly positionService: PositionService) {}

    @Get('/positions')
    async getPositions(@UserContext() user: AuthUser): Promise<PositionResponseDto[]> {
        return this.positionService.getUserPositions(user.user.id);
    }
}
