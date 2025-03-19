FROM node:22.14.0

# Make working directories
WORKDIR /app

# Upgrade pip with no cache
RUN npm i -g pnpm

# Copy package.json and pnpm-lock.yaml before installing dependencies
COPY package.json pnpm-lock.yaml ./

# Install dependencies using pnpm
RUN pnpm install --frozen-lockfile

# Copy every file in the source folder to the created working directory
COPY  . .

# Build the NestJS application
RUN pnpm build

# Run the python application
CMD ["pnpm", "start:prod"]