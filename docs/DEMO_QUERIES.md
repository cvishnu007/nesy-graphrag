# Demo Queries

## 🚫 Invalid queries (any mode)

Expect: "Not a valid query" with 0 papers, contradictions or hypotheses, and the reason. A query gate runs before every mode.

- hello how are you
- best pizza near me
- what is the weather today

## 📚 Literature Review

Expect: a cited summary of the top papers, with every retrieved paper listed below it.

- scalable training of graph neural networks on large graphs
- serverless computing performance and cold starts
- hallucination in large language models
- fairness and bias mitigation in machine learning models
- phishing website detection
- anomaly detection in multivariate sensor data
- natural language interfaces to relational databases
- continual learning and catastrophic forgetting
- traffic engineering in software-defined networks
- deep learning for network intrusion detection

## ⚡ Contradiction Detection

Expect: pairs of papers where one disputes the other, with the two conflicting sentences side by side.

- representation learning for Stack Overflow posts (a real contradiction about Post2Vec)
- low-dimensional embeddings of complex networks (a real contradiction about low-rank embeddings)
- LLM-based agents for software engineering
- benchmarking bias mitigation methods
- fairness in graph neural networks
- benchmarking heterogeneous graph neural networks
- self-training and confidence calibration in graph neural networks
- graph foundation models
- evaluation of multimodal large language models
- out-of-core and multi-GPU training of graph neural networks

## 💡 Hypothesis Generation

Expect: "method → task" ideas, i.e. methods not yet combined with the topic's task, plus an example paper. The first query takes about a minute.

- LLM-based unit test generation
- natural language interfaces to relational databases
- automated program repair
- entity resolution and record linkage
- graph neural networks for combinatorial optimization
- task scheduling in cloud computing
- query optimization with machine learning
- time series forecasting with deep learning

## 🔗 Related Work

Expect: a citation map of 10 suggested papers, each with why it was suggested and the papers that cite it.

- LLM-based agents for software engineering
- LLM-based unit test generation
- graph neural networks for combinatorial optimization
- graph neural networks for fraud detection
- fairness in graph neural networks
- retrieval-augmented generation for question answering
- vision transformers for image classification
- serverless computing performance and cold starts
