"""
Core RAG (Retrieval-Augmented Generation) engine implementation using llama_index.
This module provides the main RAG functionality for the Bank Customer Support app:
- Document ingestion from FAQ text files
- Vector indexing of documents
- Query processing using GPT-4o with retrieval augmentation

The engine uses OpenAI's API and can be configured through environment variables
for model selection and temperature settings.
"""

# pyright: reportMissingImports=false
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
# from llama_index.core.llms import ChatMessage
import os
from typing import Tuple, List
from pathlib import Path


class RAGEngine:
    """
    Core RAG (Retrieval-Augmented Generation) engine using llama_index.
    This class handles document ingestion, indexing, and query processing.
    """

    temperature = 0.2
    model = "gpt-4o"
    bank_index: VectorStoreIndex = None

    def __init__(self):
        """
        Initialize the RAG engine by setting up the document index.
        """
        # Load OpenAI API key from environment variable
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.2))

        # Initialize the LLM
        Settings.llm = OpenAI(temperature=self.temperature, model=self.model)

    def refresh_index(self):
        """
        (Re)build the document index by processing all documents in the data directory.
        """
        # Get the path relative to the current file
        bank_docs = SimpleDirectoryReader(
            input_files=[str(Path(__file__).parent / "data" / "acme_bank_faq.txt")]
        ).load_data()

        self.bank_index = VectorStoreIndex.from_documents(bank_docs)

    async def process_query(
        self, msg_list: List[Tuple[str, bool]]
    ) -> Tuple[str, List[str]]:
        """
        Process a user query using RAG with the provided chat history.

        Args:
            msg_list (List[Tuple[str, bool]]): A list of messages from session history.
                Each tuple contains:
                - str: The message text.
                - bool: Indicator if the message is from the user (True) or AI (False).

        Returns:
            Tuple[str, List[str]]: A tuple containing:
                - str: The AI's response to the user's query.
                - List[str]: A list of source documents used in generating the response.
        """
        try:
            # Build index if not already loaded
            if not self.bank_index:
                self.refresh_index()

            bank_engine = self.bank_index.as_query_engine(similarity_top_k=3)
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=bank_engine,
                    metadata=ToolMetadata(
                        name="acme_bank_faq",
                        description=(
                            "Provides FAQ information about Acme Bank"
                        ),
                    ),
                ),
                # You can add more tools here
            ]

            s_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=query_engine_tools
            )
            response = await s_engine.aquery(msg_list[-1][0])  # last user message
            return str(response), []
        except Exception as e:
            print(e)
            return "Error processing request. Try again.", []
