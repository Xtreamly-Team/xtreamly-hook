import { Module } from '@nestjs/common';
import { XtreamlyApiService } from './xtreamly-api/xtreamly-api.service';

@Module({
  providers: [XtreamlyApiService]
})
export class XtreamlyApiModule {}
