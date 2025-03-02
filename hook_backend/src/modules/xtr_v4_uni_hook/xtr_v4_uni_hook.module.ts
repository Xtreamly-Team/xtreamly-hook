import { Module } from '@nestjs/common';
import { XtrV4UniHookService } from './xtr-v4-uni-hook/xtr-v4-uni-hook.service';

@Module({
  providers: [XtrV4UniHookService]
})
export class XtrV4UniHookModule {}
