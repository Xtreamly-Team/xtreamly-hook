import { Module } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';
import { HeartbeatService } from './heartbeat.service';

@Module({
  imports: [ScheduleModule.forRoot()],
  providers: [HeartbeatService],
  exports: [HeartbeatService],
})
export class HeartbeatModule {} 