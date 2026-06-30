"""Prompt templates for synthetic data generation."""

from string import Template


class PromptTemplateLibrary:
    """Library of prompt templates for different generation tasks."""

    # RAG QA Generation
    RAG_QA_SYSTEM = """You are an expert at creating question-answer pairs for training retrieval-augmented generation systems.

Given a context passage, generate a question that can be answered using the information in the context, along with a comprehensive answer.

Requirements:
- Question should be natural and specific
- Answer should be accurate and based on the context
- Answer should be self-contained (understandable without the context)
- Vary question types: factual, analytical, comparative, etc.
"""

    RAG_QA_USER = Template("""Context:
$context

Difficulty level: $difficulty

Generate a question-answer pair at this difficulty level.
- Easy: Simple factual questions with direct answers
- Medium: Questions requiring understanding and synthesis
- Hard: Questions requiring inference or combining multiple facts
- Expert: Complex analytical questions

Output format (JSON):
{
    "question": "...",
    "answer": "...",
    "reasoning": "Why this question is at the specified difficulty"
}""")

    # Instruction Generation
    INSTRUCTION_SYSTEM = Template("""You are an expert at creating instruction-following examples for training language models.

Generate diverse, realistic instructions with corresponding inputs and outputs. The instructions should be clear and the outputs should be high-quality.

Focus on task type: $task_type
""")

    INSTRUCTION_USER = Template("""Generate an instruction-following example.

Task type: $task_type
Difficulty: $difficulty
Domain: $domain

Requirements:
- Instruction should be clear and actionable
- Input should be realistic
- Output should be comprehensive and correct
- Follow the specified difficulty level

Output format (JSON):
{
    "instruction": "...",
    "input": "...",
    "output": "...",
    "explanation": "Brief explanation of why output is correct"
}""")

    # Conversation Generation
    CONVERSATION_SYSTEM = """You are an expert at creating multi-turn conversations for training conversational AI.

Generate natural, coherent conversations that demonstrate helpful assistant behavior.
"""

    CONVERSATION_USER = Template("""Generate a multi-turn conversation.

Topic: $topic
Number of turns: $num_turns
Difficulty: $difficulty
Domain: $domain

Requirements:
- Conversation should be natural and coherent
- Each turn should build on previous context
- Assistant responses should be helpful and informative
- Include appropriate follow-up questions

Output format (JSON):
{
    "system_prompt": "Optional system prompt for the assistant",
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}""")

    # Question from Answer (Reverse)
    REVERSE_QA_SYSTEM = """You are an expert at creating questions that would lead to a given answer.

Given an answer and context, generate a natural question that would elicit this answer.
"""

    REVERSE_QA_USER = Template("""Answer: $answer

Context: $context

Generate a question that would naturally lead to this answer.

Output format (JSON):
{
    "question": "...",
    "question_type": "factual/analytical/comparative/etc."
}""")

    # Paraphrase for Augmentation
    PARAPHRASE_SYSTEM = """You are an expert at paraphrasing text while preserving meaning.

Generate paraphrases that maintain semantic equivalence but vary in structure and vocabulary.
"""

    PARAPHRASE_USER = Template("""Original text: $text

Generate $num_paraphrases paraphrases with varying styles:
- Formal
- Casual
- Concise
- Detailed

Output format (JSON):
{
    "paraphrases": [
        {"text": "...", "style": "..."}
    ]
}""")


class DomainPromptTemplates:
    """Domain-specific prompt templates and configurations."""

    LEGAL = {
        "system_context": """You are a legal expert. Generate examples using proper legal terminology and considering legal principles. Be precise with citations and legal concepts.""",
        "requirements": [
            "Use appropriate legal terminology",
            "Reference relevant laws or precedents when applicable",
            "Maintain formal tone",
            "Consider jurisdictional differences",
        ],
        "disclaimer": "This is for educational purposes only and does not constitute legal advice.",
    }

    MEDICAL = {
        "system_context": """You are a medical expert. Generate examples using proper medical terminology and following evidence-based medicine principles. Always include appropriate disclaimers.""",
        "requirements": [
            "Use correct medical terminology",
            "Reference clinical guidelines when applicable",
            "Include appropriate safety considerations",
            "Note when professional consultation is advised",
        ],
        "disclaimer": "This information is for educational purposes only. Consult a healthcare provider for medical advice.",
    }

    TECHNICAL = {
        "system_context": """You are a technical documentation expert. Generate examples that are precise, well-structured, and follow technical writing best practices.""",
        "requirements": [
            "Use precise technical terminology",
            "Include code examples when relevant",
            "Follow consistent formatting",
            "Provide clear explanations",
        ],
        "disclaimer": None,
    }

    FINANCIAL = {
        "system_context": """You are a financial expert. Generate examples using proper financial terminology and considering regulatory requirements. Include appropriate disclaimers.""",
        "requirements": [
            "Use accurate financial terminology",
            "Consider regulatory context",
            "Include risk disclosures when appropriate",
            "Note that this is not financial advice",
        ],
        "disclaimer": "This is for educational purposes only and does not constitute financial advice.",
    }

    SCIENTIFIC = {
        "system_context": """You are a scientific expert. Generate examples using proper scientific methodology and terminology. Cite sources where appropriate.""",
        "requirements": [
            "Use accurate scientific terminology",
            "Follow scientific method principles",
            "Distinguish between established facts and hypotheses",
            "Include relevant citations or references",
        ],
        "disclaimer": None,
    }

    @classmethod
    def get_domain_config(cls, domain: str) -> dict:
        """Get configuration for a specific domain."""
        domain_upper = domain.upper()
        if hasattr(cls, domain_upper):
            return getattr(cls, domain_upper)
        return {
            "system_context": f"You are an expert in {domain}.",
            "requirements": [],
            "disclaimer": None,
        }
