# LangChain Agents
Using LangChain agents to fetch data from Wikipedia, Arxiv, and a local vector database.

# Agents
Agents execute sequences of actions that utilize tools. Tools are employed to retrieve information from external sources when the LLM (Language Learning Model) lacks the required data.

For example, we are using ChatGPT 3.5, which is trained on data up to September 2021.
If we ask questions about events or information before September 2021, the LLM can provide results. However, if we ask questions about events or information beyond that date, the LLM cannot answer these queries. This is where agents/tools come into play, helping the LLM to fetch data from external sources.

## Generally, any LLM consists of two components:

Knowledge Engine - Trained on data up to a specific point in time.

Reasoning Engine - Uses agents and tools to retrieve or process external information.
