"""
Async memory extractor for Memory SDK.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from memory_sdk.config import MemoryConfig
from memory_sdk.embeddings.provider import EmbeddingProvider
from memory_sdk.extraction.strategies import DEFAULT_STRATEGIES, Strategy
from memory_sdk.models import Memory, Message, ExtractionResult

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extracts structured memories from conversations using LLMs."""
    
    def __init__(
        self,
        config: MemoryConfig,
        embedding_provider: EmbeddingProvider,
        strategies: Optional[Dict[str, Strategy]] = None,
    ):
        self.config = config
        self.embedding_provider = embedding_provider
        self.strategies = strategies or DEFAULT_STRATEGIES
        self._llm_client = None
    
    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            provider = self.config.llm.provider.lower()
            
            if provider == "openai":
                from openai import AsyncOpenAI
                self._llm_client = AsyncOpenAI(api_key=self.config.llm.openai_api_key)
            elif provider == "anthropic":
                from anthropic import AsyncAnthropic
                self._llm_client = AsyncAnthropic(api_key=self.config.llm.anthropic_api_key)
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")
        
        return self._llm_client
    
    def _build_extraction_prompt(
        self,
        messages: List[Message],
        strategy: Strategy,
        existing_memory: Optional[Memory] = None,
    ) -> str:
        """Build the extraction prompt."""
        # Format conversation
        conversation = "\n".join([
            f"{msg.role.value.upper()}: {msg.content}"
            for msg in messages
        ])
        
        # Format existing memory
        existing_str = ""
        if existing_memory:
            existing_str = f"""
EXISTING {strategy.name.upper()} DATA:
{json.dumps(existing_memory.content, indent=2)}

Update and merge this with any new information from the conversation.
"""
        
        return f"""Extract {strategy.name} information from this conversation.

STRATEGY: {strategy.name}
DESCRIPTION: {strategy.description}

EXPECTED OUTPUT SCHEMA:
{json.dumps(strategy.schema, indent=2)}
{existing_str}
CONVERSATION:
{conversation}

INSTRUCTIONS:
1. Extract ONLY {strategy.name} information that matches the schema
2. Return a valid JSON object matching the schema
3. Include a "summary" field with a plain English summary (1-2 sentences)
4. If no relevant information found, return {{"summary": null}}
5. Merge with existing data if provided

Return ONLY the JSON object, no other text."""
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM to extract memories."""
        provider = self.config.llm.provider.lower()
        
        if provider == "openai":
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": "You are a memory extraction assistant. Extract structured information from conversations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        
        elif provider == "anthropic":
            response = await self.llm_client.messages.create(
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    async def extract_for_strategy(
        self,
        user_id: str,
        session_id: str,
        messages: List[Message],
        strategy: Strategy,
        existing_memory: Optional[Memory] = None,
    ) -> Optional[Memory]:
        """Extract memory for a single strategy."""
        if not strategy.enabled:
            return None
        
        try:
            # Build prompt
            prompt = self._build_extraction_prompt(messages, strategy, existing_memory)
            
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse response
            data = json.loads(response)
            
            # Check if extraction found anything
            if data.get("summary") is None:
                return None
            
            # Extract summary
            summary = data.pop("summary", "")
            if not summary:
                # Generate summary from content
                summary = f"{strategy.name}: {json.dumps(data)[:200]}"
            
            # Generate embedding from summary
            embedding = await self.embedding_provider.embed_text(summary)
            
            # Create memory
            memory_id = existing_memory.memory_id if existing_memory else str(uuid4())
            
            return Memory(
                memory_id=memory_id,
                user_id=user_id,
                strategy=strategy.name,
                content=data,
                summary=summary,
                embedding=embedding,
                confidence=1.0,
                created_at=existing_memory.created_at if existing_memory else datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={"session_id": session_id},
            )
        
        except Exception as e:
            logger.error(f"Failed to extract {strategy.name}: {e}")
            return None
    
    async def extract_memories(
        self,
        user_id: str,
        session_id: str,
        messages: List[Message],
        existing_memories: Optional[Dict[str, Memory]] = None,
    ) -> ExtractionResult:
        """Extract memories from a conversation."""
        start_time = time.time()
        existing_memories = existing_memories or {}
        
        extracted = []
        updated = []
        strategies_processed = []
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            
            strategies_processed.append(strategy_name)
            existing = existing_memories.get(strategy_name)
            
            memory = await self.extract_for_strategy(
                user_id=user_id,
                session_id=session_id,
                messages=messages,
                strategy=strategy,
                existing_memory=existing,
            )
            
            if memory:
                if existing:
                    updated.append(memory)
                else:
                    extracted.append(memory)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            user_id=user_id,
            session_id=session_id,
            memories_extracted=len(extracted),
            memories_updated=len(updated),
            strategies_processed=strategies_processed,
            duration_ms=duration_ms,
        ), extracted + updated
    
    def build_context_string(self, memories: List[Memory]) -> str:
        """Build a context string from memories."""
        if not memories:
            return ""
        
        summaries = [m.summary for m in memories if m.summary]
        return " ".join(summaries)

