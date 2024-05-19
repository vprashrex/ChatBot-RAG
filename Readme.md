# ChatBot: RAG BASED SLM FOR Python GEN-AI Developer

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)


## Introduction

Project: ChatBot: RAG BASED SLM FOR Python GEN-AI Developer

## Endpoint that Server Runs On

- `http://localhost:8000/`
- `http://localhost:8000/chat`

## Getting Started

### Installation

To install the necessary dependencies, follow these steps:

1. Clone the project:
    ```sh
    git clone https://github.com/vprashrex/ChatBot-RAG.git
    ```

2. Navigate to the project directory:
    ```sh
    cd ChatBot-RAG
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the server using:
    ```sh
    uvicorn server:app
    ```

### Usage

After running the server, use the following endpoints:

- To view the Citations output, go to `http://localhost:8000`
- To chat with Mock LLM and see how the RAG Component works, go to `http://localhost:8000/chat`

**NOTE:** Wait for a few minutes before visiting the URLs, as all the dependencies and files will be downloading.

### ChatBot: RAG BASED LLAM2-7B-GGUF

Below is the Colab link to demonstrate how RAG is used in conjunction with an LLM to provide accurate information based on `data.json`:

[Colab Link](https://colab.research.google.com/drive/1vwDMwD5S8oPyZuW1f8tPJyeMj1lzPFr0?usp=sharing)

### Demo Video

I also included a demo video so that you can understand how Retrieved Augmented Generation (RAG) works and how it will be useful in the chatbot. To retrive accurate result based on user query