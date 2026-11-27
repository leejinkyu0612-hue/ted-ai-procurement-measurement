# -*- coding: utf-8 -*-
"""Fixed prompts (pre-specified, 2026-11-27)."""

ANNOTATION_PROMPT = """You are annotating public procurement award notices for a research codebook.

Definition. A notice is AI-related (ai = 1) if the procured object includes the development, supply, licensing, operation, or consulting/training on a system or service whose core functionality relies on artificial-intelligence techniques: machine learning, deep learning, natural language processing or large language models, chatbots or conversational agents, computer vision or image recognition, speech recognition or synthesis, or predictive models learned from data, including automation that the notice explicitly describes as AI-based.
A notice is NOT AI-related (ai = 0) if it buys generic software, ERP, websites, RPA or workflow automation without AI, BI dashboards or data warehouses without learning components, "smart" devices without AI functionality, or hardware only.
Judge only from the text. The text may be in any EU language.

Notice text:
\"\"\"{text}\"\"\"

Answer with JSON only: {{"ai": 0}} or {{"ai": 1}}"""

NLI_HYPOTHESIS = "This procurement is about artificial intelligence."

PROTOTYPES = [
    "query: procurement of artificial intelligence software or AI services",
    "query: development of a machine learning or deep learning system",
    "query: chatbot, virtual assistant or large language model solution",
    "query: computer vision, image recognition or speech recognition system",
    "query: predictive analytics model trained on data using AI",
]
