FROM ubuntu:22.04

WORKDIR /app

# Install Node.js and npm
RUN apt-get update && \
    apt-get install -y nodejs npm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
