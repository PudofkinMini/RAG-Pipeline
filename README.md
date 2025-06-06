# RAG Pipeline

This repository contains a simple Retrieval-Augmented Generation (RAG) pipeline. It is organized into two main parts:

- **frontend** – a single page application built with React, Vite and TypeScript styled with Tailwind CSS.
- **backend** – a lightweight Flask server that will expose API endpoints used by the frontend.

The goal of this project is to experiment with using a RAG approach to help readers better understand Canadian bills. The pipeline will process legislative texts and provide easy-to-digest summaries and references in the web application.

## Getting Started

1. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   This starts the development server on `localhost:5173`.

2. **Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```
   The Flask server will run on `localhost:5000`.

With both servers running, the React app can call the Flask API to retrieve data from the RAG pipeline.
