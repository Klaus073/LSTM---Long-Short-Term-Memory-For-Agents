"""
Async memory extractor for Memory SDK.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from memory_sdk.config import MemoryConfig
from memory_sdk.embeddings.provider import EmbeddingProvider
from memory_sdk.models import Memory, Message, ExtractionResult

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extracts user information from conversations using LLMs."""
    
    # System prompt for memory extraction
    SYSTEM_PROMPT = """
##Your Role##
You are user's long-term memory assistant. Your job is to build and maintain a user's long-term memory.

##YOUR TASK##
- Extract user's information from the RECENT SESSION and create/update the user's long-term memory.

#CRITICAL RULES#
1. If EXISTING LONG-TERM MEMORY is empty: Create new memory from the session
2. If EXISTING LONG-TERM MEMORY exists: UPDATE and APPEND new information
3. NEVER REMOVE existing information - only add or update
4. If a field has new value, update it (e.g., job changed from "Engineer" to "Senior Engineer")
5. For list items: APPEND new items, keep existing ones
6. Output in clean Markdown format (see OUTPUT FORMAT below)

##Key Points - Personify the User##
Extract everything that helps personify the user:
- Name, job, company, location
- Likes, dislikes, preferences
- Activities, interests, hobbies
- Goals, dreams, aspirations
- Fears, challenges, struggles
- Strengths, weaknesses
- Values, beliefs
- Habits, routines
- Important facts
- Relationships, family
- Any other personal information

##OUTPUT FORMAT##
Return ONLY the user memory in this Markdown format (omit sections with no info):

## About ##
Personify the user. get its name, job, company, location, etc. all teh personal details realted to him.

## Interests & Preferences
- [Interest/preference 1]
- [Interest/preference 2]

## Facts
- [Fact 1]
- [Fact 2]

## Goals & Aspirations
- [Goal 1]

## Other
- [Any other relevant info]"""

    def __init__(
        self,
        config: MemoryConfig,
        embedding_provider: EmbeddingProvider,
    ):
        self.config = config
        self.embedding_provider = embedding_provider
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
        existing_memory: Optional[str] = None,
    ) -> str:
        """Build the user prompt with task-specific data."""
        # Format conversation
        conversation = "\n".join([
            f"{msg.role.value.upper()}: {msg.content}"
            for msg in messages
        ])
        
        # Format existing memory (already Markdown string)
        if existing_memory:
            existing_section = f"""#EXISTING LONG-TERM MEMORY#
{existing_memory}"""
        else:
            existing_section = """#EXISTING LONG-TERM MEMORY#
None - This is a new user with no saved memory yet."""
        
        return f"""{existing_section}

#RECENT SESSION (Just Happened)#
{conversation}

Now output the updated user memory in Markdown format."""
    
    async def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM to extract memories."""
        provider = self.config.llm.provider.lower()
        
        if provider == "openai":
            response = await self.llm_client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            return response.choices[0].message.content
        
        elif provider == "anthropic":
            response = await self.llm_client.messages.create(
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    async def extract_memories(
        self,
        user_id: str,
        session_id: str,
        messages: List[Message],
        existing_memory: Optional[str] = None,
    ) -> tuple[ExtractionResult, Optional[Memory]]:
        """
        Extract user memory from a conversation.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            messages: List of messages from the session
            existing_memory: Existing LTM as Markdown string (or None for new user)
            
        Returns:
            Tuple of (ExtractionResult, Memory or None)
        """
        start_time = time.time()
        
        if not messages:
            return ExtractionResult(
                user_id=user_id,
                session_id=session_id,
                memories_extracted=0,
                memories_updated=0,
                strategies_processed=[],
                duration_ms=0,
            ), None
        
        try:
            # Build prompt
            prompt = self._build_extraction_prompt(messages, existing_memory)
            
            # Call LLM - returns Markdown string
            ltm_content = await self._call_llm(prompt)
            ltm_content = ltm_content.strip()
            
            # Generate embedding from the content
            embedding = await self.embedding_provider.embed_text(ltm_content)
            
            # Create single memory object
            is_update = existing_memory is not None
            
            memory = Memory(
                memory_id=str(uuid4()),
                user_id=user_id,
                strategy="user_memory",
                content=ltm_content,  # Store as string directly
                summary=ltm_content,   # Same as content for MD format
                embedding=embedding,
                confidence=1.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={"session_id": session_id},
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ExtractionResult(
                user_id=user_id,
                session_id=session_id,
                memories_extracted=0 if is_update else 1,
                memories_updated=1 if is_update else 0,
                strategies_processed=["user_memory"],
                duration_ms=duration_ms,
            ), memory
        
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return ExtractionResult(
                user_id=user_id,
                session_id=session_id,
                memories_extracted=0,
                memories_updated=0,
                strategies_processed=[],
                duration_ms=(time.time() - start_time) * 1000,
            ), None
    
    def build_context_string(self, memory: Optional[Memory]) -> str:
        """Build a context string from memory."""
        if not memory:
            return ""
        return memory.summary or ""
