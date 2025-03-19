# Xtreamly Hook - AI Backend

Welcome to the Xtreamly AI Backend.
This backend is providing the intelligent hedging strategy given a user and their currently active positions, utilizing Xtreamly's volatility models.

Read more about our volatility prediction model here: [AI Volatility by Xtreamly.pdf](docs%2FAI%20Volatility%20by%20Xtreamly.pdf)

# Prerequisites

1. Python 3.11

## 🛠 Local setup

1. Install the required python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Setup `.env` variables as in the `.env.example

   ```bash
   cp .env.example .env
   ```

## 🚀 Run Locally

1. Start the backend agent server:

   ```bash
   python main.py
   ```

   By default, the server runs on port `8001`. This can be customized using the `PORT` environment variable.

   Visit API: http://localhost:8001

## 💻 Build Docker Image

   ```bash
   docker build -t xtreamly-ai-api .
   ```
### Push docker image to docker hub:

```bash
docker tag xtreamly-ai-api xtreamly/xtreamly-ai-api:latest
docker push xtreamly/xtreamly-ai-api:latest
```

## Knowledge

- https://www.youtube.com/watch?v=dW-qr_ntOgc&t=173s
- https://fastapi.tiangolo.com/advanced/websockets/
