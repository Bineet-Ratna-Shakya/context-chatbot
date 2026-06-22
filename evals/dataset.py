# Labeled set for evaluating retrieval and answer quality against the committed
# corpus in docs/. `expected_source` is matched against the titles of retrieved
# sources; `expected_keywords` are matched case-insensitively against the answer.
# A None `expected_source` is an off-topic question: retrieval should return
# nothing and the bot should abstain.

DATASET = [
    {
        "question": "What is LangGraph?",
        "expected_source": "LangGraph Overview",
        "expected_keywords": ["stateful", "LangChain"],
    },
    {
        "question": "Who develops LangGraph?",
        "expected_source": "LangGraph Overview",
        "expected_keywords": ["LangChain"],
    },
    {
        "question": "What is retrieval-augmented generation?",
        "expected_source": "Retrieval-augmented generation",
        "expected_keywords": ["retrieve", "external"],
    },
    {
        "question": "What is LangChain?",
        "expected_source": "LangChain",
        "expected_keywords": ["framework"],
    },
    {
        "question": "Which vector database does this chatbot use?",
        "expected_source": "Context Chatbot Architecture",
        "expected_keywords": ["Chroma"],
    },
    {
        "question": "How do I bake sourdough bread?",
        "expected_source": None,
        "expected_keywords": [],
    },
]
