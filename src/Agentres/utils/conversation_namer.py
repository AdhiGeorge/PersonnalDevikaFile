from typing import List, Dict, Any
from Agentres.llm.llm import LLM

class ConversationNamer:
    def __init__(self):
        self.llm = LLM()

    def generate_name(self, query: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a descriptive name for a conversation based on the query and context."""
        # Prepare the prompt
        prompt = self._create_prompt(query, context)
        
        # Get response from LLM
        response = self.llm.generate(prompt, max_tokens=30)
        
        # Clean and format the response
        name = self._clean_name(response)
        
        return name

    def _create_prompt(self, query: str, context: List[Dict[str, Any]] = None) -> str:
        """Create a prompt for the LLM to generate a conversation name."""
        base_prompt = (
            "Generate a brief, descriptive name (3-5 words) for a conversation that starts with this query. "
            "The name should capture the main topic or goal of the conversation. "
            "Format: Return only the name, no additional text or punctuation.\n\n"
            f"Query: {query}\n"
        )
        
        if context:
            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context[-3:]])
            base_prompt += f"\nRecent context:\n{context_str}\n"
        
        return base_prompt

    def _clean_name(self, name: str) -> str:
        """Clean and format the generated name."""
        # Remove any quotes, periods, or extra whitespace
        name = name.strip('"\'., ')
        
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        # Ensure the name is not too long
        words = name.split()
        if len(words) > 5:
            name = ' '.join(words[:5])
        
        return name

    def update_name(self, conversation_id: int, new_context: List[Dict[str, Any]]) -> str:
        """Update an existing conversation name based on new context."""
        # Extract the original query from context
        original_query = next((msg['content'] for msg in new_context if msg['role'] == 'user'), None)
        
        if original_query:
            return self.generate_name(original_query, new_context)
        return None 