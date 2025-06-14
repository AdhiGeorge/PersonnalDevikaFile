import openai
import asyncio
import os
from src.config import Config
from src.logger import Logger

log = Logger()

class AzureOpenAI:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_API_VERSION_CHAT", "2024-02-15-preview")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        
        if not self.api_key or not self.api_base:
            raise ValueError("Azure OpenAI API key and endpoint must be configured in environment variables")
        
        try:
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base
            )
            log.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    async def inference(self, model_id: str, prompt: str) -> str:
        try:
            loop = asyncio.get_event_loop()
            chat_completion = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.deployment_name,  # Use the deployment name instead of model_id
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.strip(),
                        }
                    ],
                    temperature=0
                )
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            log.error(f"Error during Azure OpenAI inference: {str(e)}")
            raise 