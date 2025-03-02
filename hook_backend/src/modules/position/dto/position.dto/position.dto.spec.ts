import { OpenPositionQuoteDto, PositionResponseDto, PositionHistoryDto, UpdatePositionStatusDto, RecordPositionHistoryDto } from './position.dto';

describe('Position', () => {
  it('should be defined', () => {
    expect(new OpenPositionQuoteDto()).toBeDefined();
    expect(new PositionResponseDto()).toBeDefined();
    expect(new PositionHistoryDto()).toBeDefined();
    expect(new UpdatePositionStatusDto()).toBeDefined();
    expect(new RecordPositionHistoryDto()).toBeDefined();
  });
});
