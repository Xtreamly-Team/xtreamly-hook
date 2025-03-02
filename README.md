# hook_backend - DeFi Backend System

A NestJS backend system that integrates with Uniswap V4 and GMX to provide users with hedged positions.

## Technology Stack

- **Backend**: NestJS with TypeScript
- **Database**: PostgreSQL with TimescaleDB extension
- **ORM**: TypeORM
- **Blockchain Integration**: viem
- **Microservices**: Integration with Python GMX hedging service

## Features

- User registration
- Position management
- Automated hedging on GMX
- Integration with Uniswap V4 hooks
- Time-series data storage for position history

## Prerequisites

- Node.js LTS (v22.x)
- Docker and Docker Compose
- PNPM (for package management)

## Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd hook_backend
```

2. **Install dependencies**

```bash
pnpm install
```

3. **Create .env file**

Copy the example `.env` file and adjust as needed:

```bash
cp .env.example .env
```

4. **Start the database**

```bash
docker-compose up -d xtr_trade_db
```

5. **Run migrations**

```bash
pnpm run typeorm:run-migrations
```

6. **Start the application**

```bash
pnpm run start:dev
```

The application will be available at http://localhost:3000

## Development

### Generate a migration

```bash
pnpm run typeorm:generate-migration src/modules/database/migrations/MigrationName
```

### Create a new migration

```bash
pnpm run typeorm:create-migration src/modules/database/migrations/MigrationName
```

### Run migrations

```bash
pnpm run typeorm:run-migrations
```

### Revert the last migration

```bash
pnpm run typeorm:revert-migration
```

## API Endpoints

### User Management

- `POST /users/register` - Register a new user
- `GET /users/:walletAddress` - Get user by wallet address

### Position Management

- `POST /positions/quote` - Get a quote for opening a position
- `POST /positions` - Open a new position
- `GET /positions/user/:userId` - Get positions for a user
- `GET /positions/:id` - Get position details
- `GET /positions/:id/history` - Get position history
- `PATCH /positions/:id/status` - Update position status

## Database Schema

### Users Table

- `id` - UUID primary key
- `walletAddress` - User's blockchain wallet address
- `email` - Optional email address
- `isActive` - User account status
- `createdAt` - Creation timestamp
- `updatedAt` - Last update timestamp

### Positions Table

- `id` - UUID primary key
- `userId` - Foreign key to users table
- `tokenA` - First token address
- `tokenB` - Second token address
- `amountA` - Amount of first token
- `amountB` - Amount of second token
- `lowerTick` - Lower tick price bound
- `upperTick` - Upper tick price bound
- `hedgeAmount` - Amount hedged on GMX
- `status` - Position status (pending/active/rebalancing/closed)
- `uniswapPositionId` - Uniswap position identifier
- `gmxPositionId` - GMX position identifier
- `metadata` - Additional position metadata
- `createdAt` - Creation timestamp
- `updatedAt` - Last update timestamp

### Position History Table (TimescaleDB hypertable)

- `id` - UUID primary key
- `positionId` - Foreign key to positions table
- `tokenAValue` - Value of token A at this point in time
- `tokenBValue` - Value of token B at this point in time
- `hedgeValue` - Value of the hedge position
- `netValue` - Net position value
- `metadata` - Additional historical metadata
- `timestamp` - Time of the history record (TimescaleDB partition column)