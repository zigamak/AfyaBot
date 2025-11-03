import logging
import os
import re
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class AIService:
    """
    Handles AI-powered conversational interactions for a gynecology medical chatbot,
    focusing on cervical screening, reproductive health, and community health worker integration.
    """

    def __init__(self, config, data_manager):
        load_dotenv()
        self.data_manager = data_manager

        # Initialize configuration
        try:
            if isinstance(config, dict):
                self.supabase_url = config.get("supabase_url")
                self.supabase_key = config.get("supabase_service_key")
                self.openai_api_key = config.get("openai_api_key")
            else:
                self.supabase_url = getattr(config, 'SUPABASE_URL', None)
                self.supabase_key = getattr(config, 'SUPABASE_SERVICE_KEY', None)
                self.openai_api_key = getattr(config, 'OPENAI_API_KEY', None)

            # Validate configuration
            self.ai_enabled = bool(self.supabase_url and self.supabase_key and self.openai_api_key)
            logger.info(f"Configuration check - Supabase URL: {'set' if self.supabase_url else 'missing'}")
            logger.info(f"Configuration check - Supabase Key: {'set' if self.supabase_key else 'missing'}")
            logger.info(f"Configuration check - OpenAI API Key: {'set' if self.openai_api_key else 'missing'}")
            logger.info(f"AI Enabled: {self.ai_enabled}")

            # Initialize Supabase client and RAG pipeline
            self.supabase_client = None
            self.vector_store = None
            self.agent_executor = None
            self.llm = None
            
            if self.ai_enabled:
                # Initialize Supabase client
                try:
                    self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                    logger.info("Supabase client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
                    self.ai_enabled = False
                    return

                # Initialize embeddings and vector store
                try:
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=self.openai_api_key)
                    self.vector_store = SupabaseVectorStore(
                        embedding=embeddings,
                        client=self.supabase_client,
                        table_name="gynecology_documents",
                        query_name="match_documents",
                    )
                    logger.info("Vector store initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
                    self.ai_enabled = False
                    return

                # Initialize LLM and agent
                try:
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=self.openai_api_key)
                    
                    tools = [self._create_retrieval_tool()]
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", self._get_system_prompt()),
                        MessagesPlaceholder(variable_name="chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])

                    agent = create_tool_calling_agent(self.llm, tools, prompt)
                    self.agent_executor = AgentExecutor(
                        agent=agent, 
                        tools=tools, 
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=3
                    )
                    logger.info("RAG pipeline initialized successfully for Gynecology Medical AI.")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM or agent: {e}", exc_info=True)
                    self.ai_enabled = False
                    return
            else:
                logger.warning("AI features disabled - missing Supabase or OpenAI configuration.")
        except Exception as e:
            logger.error(f"Unexpected error during AIService initialization: {e}", exc_info=True)
            self.ai_enabled = False

    def _create_retrieval_tool(self):
        """Create the retrieval tool with proper access to vector store."""
        vector_store = self.vector_store
        
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve medical information related to gynecology and cervical screening from the knowledge base."""
            try:
                logger.info(f"Retrieving documents for query: {query}")
                # Use similarity_search with explicit limit parameter
                retrieved_docs = vector_store.similarity_search(query, k=3)
                
                if not retrieved_docs:
                    logger.warning(f"No documents retrieved for query: {query}")
                    return "No relevant information found in the knowledge base.", []
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs
            except Exception as e:
                logger.error(f"Error in retrieval tool: {e}", exc_info=True)
                return f"Error retrieving information: {str(e)}", []
        
        return retrieve

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Gynecology Medical AI."""
        return """You are AfyaBot, a helpful and empathetic medical assistant specializing in gynecology and women's reproductive health.

Your Role:
- Answer questions about cervical screening, reproductive health, and women's wellness
- Provide accurate information based on the retrieved medical documents
- Be warm, supportive, and use simple, clear language
- Encourage users to seek professional medical care when appropriate

Guidelines:
1. ALWAYS use the retrieve tool to search for relevant information before answering
2. Base your answers ONLY on the retrieved medical documents
3. If information isn't found in the documents, say: "I don't have specific information about that in my knowledge base. Please consult a qualified healthcare professional."
4. Keep answers concise but informative (2-4 sentences for most questions)
5. Use emojis sparingly (🌸, 💙, ✅) to make responses friendly
6. Never diagnose conditions or prescribe treatments
7. For urgent concerns, always recommend contacting a healthcare provider immediately
8. If the user asks about kit requests or test results, acknowledge their request but remind them to follow the proper channels

Important:
- You MUST call the retrieve tool for medical questions
- Never make up medical information
- Be culturally sensitive and respectful
- Maintain patient confidentiality

Remember: You're here to inform and support, not to replace professional medical care."""

    def _is_swahili(self, message: str) -> bool:
        """Detect if the message is in Swahili based on common words."""
        if not message or not isinstance(message, str):
            return False
        swahili_keywords = [
            r'ninaweza', r'je\b', r'wapi', r'gani', r'kifaa', r'kupima',
            r'afya', r'mwanamke', r'shukrani', r'tafadhali', r'habari',
            r'nini', r'vipi', r'kwa\s', r'ya\s', r'na\s'
        ]
        message = message.lower().strip()
        matches = sum(1 for pattern in swahili_keywords if re.search(pattern, message, re.IGNORECASE))
        return matches >= 2

    def _is_kit_request(self, message: str) -> bool:
        """Check if the user message indicates a request to order a screening kit."""
        if not message or not isinstance(message, str):
            return False
        kit_keywords = [
            r'order.*kit', r'request.*screening', r'get.*kit', r'need.*kit',
            r'want.*kit', r'screening.*kit', r'test.*kit',
            r'kifaa.*kupima', r'pata.*kifaa', r'omba.*kifaa', r'nataka.*kifaa'
        ]
        return any(re.search(pattern, message.lower(), re.IGNORECASE) for pattern in kit_keywords)

    def _is_result_request(self, message: str) -> bool:
        """Check if the user message is requesting test results."""
        if not message or not isinstance(message, str):
            return False
        result_keywords = [
            r'\bresult', r'test result', r'my result', r'check result',
            r'outcome', r'pima.*matokeo', r'\bmatokeo', r'check.*result',
            r'show.*result', r'view.*result'
        ]
        return any(re.search(pattern, message.lower(), re.IGNORECASE) for pattern in result_keywords)

    def _extract_email(self, message: str) -> str:
        """Extract email from user message if present."""
        if not message or not isinstance(message, str):
            return None
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
        return email_match.group(0) if email_match else None

    def _extract_name(self, message: str) -> str:
        """Extract a name from user message if present (basic heuristic)."""
        if not message or not isinstance(message, str):
            return None
        name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', message)
        return name_match.group(0) if name_match else None

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        if not email or not isinstance(email, str):
            return False
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(email_pattern, email))

    def _extract_location(self, message: str) -> str:
        """Extract location from user message (e.g., county or city)."""
        if not message or not isinstance(message, str):
            return None
        location_match = re.search(
            r'\b(Nairobi|Mombasa|Kisumu|Nakuru|Eldoret|Thika|Malindi|Kitale|'
            r'Garissa|Kakamega|Nyeri|Machakos|Meru|Embu|Kericho|Kisii|'
            r'Kilifi|Kwale|Lamu|Taita|Taveta|Bungoma|Busia|Siaya|Migori|'
            r'Homa Bay|Narok|Kajiado|Turkana|West Pokot|Samburu|'
            r'Trans Nzoia|Uasin Gishu|Elgeyo Marakwet|Nandi|Baringo|'
            r'Laikipia|Nyamira|Vihiga|Bomet|Makueni|Kitui|Tharaka|Nithi|'
            r'Marsabit|Isiolo|Mandera|Wajir|Tana River|Murang\'?a|Kiambu|'
            r'Nyandarua|Kirinyaga|County)\b',
            message, 
            re.IGNORECASE
        )
        return location_match.group(0) if location_match else None

    def process_kit_request(self, user_message: str, phone_number: str, user_name: str = None, session_id: str = None) -> tuple[str, bool, str, str]:
        """
        Process a screening kit request, checking for location and name.
        """
        if not user_message or not phone_number:
            logger.error(f"Invalid input: user_message={user_message}, phone_number={phone_number}")
            response = "🤖 Sorry, I couldn't process your request. Please provide valid input."
            if self._is_swahili(user_message):
                response = "🤖 Samahani, sikuweza kushughulikia ombi lako. Tafadhali toa maelezo sahihi."
            return response, False, None, None

        location = self._extract_location(user_message)
        name = user_name or self._extract_name(user_message)

        if location and name:
            try:
                response = (
                    f"Thank you, {name}! Your request for a cervical screening kit in {location} has been received. 🎉\n"
                    "A Community Health Worker will contact you soon to arrange delivery. Anything else I can help with?"
                )
                if self._is_swahili(user_message):
                    response = (
                        f"Asante, {name}! Ombi lako la kifaa cha uchunguzi wa saratani ya shingo ya kizazi huko {location} limepokewa. 🎉\n"
                        "Mhudumu wa Afya ya Jamii atakupigia simu hivi karibuni kupanga utoaji. Je, kuna jambo lingine laweza kukusaidia?"
                    )
                return response, False, location, name
            except Exception as e:
                logger.error(f"Error processing kit request for {phone_number}: {e}", exc_info=True)
                response = (
                    "🤖 Sorry, I couldn't process your kit request right now. Please try again or contact a health worker."
                )
                if self._is_swahili(user_message):
                    response = (
                        "🤖 Samahani, sikuweza kushughulikia ombi lako la kifaa sasa hivi. Tafadhali jaribu tena au wasiliana na mhudumu wa afya."
                    )
                return response, False, None, None
        else:
            missing = []
            if not location:
                missing.append("your area or county")
            if not name:
                missing.append("your name")
            response = (
                f"Thank you for your interest in a cervical screening kit! 🎉\n"
                f"Please provide {' and '.join(missing)} to complete your request."
            )
            if self._is_swahili(user_message):
                missing_sw = []
                if not location:
                    missing_sw.append("eneo lako au kaunti")
                if not name:
                    missing_sw.append("jina lako")
                response = (
                    f"Asante kwa nia yako ya kupata kifaa cha uchunguzi wa saratani ya shingo ya kizazi! 🎉\n"
                    f"Tafadhali toa {' na '.join(missing_sw)} ili kukamilisha ombi lako."
                )
            logger.info(f"Prompting for missing kit request info: {missing} for phone_number={phone_number}")
            return response, True, location, name

    def process_result_request(self, user_message: str, phone_number: str, user_name: str = None, session_id: str = None) -> str:
        """
        Process a request for test results via AFYA KE system.
        """
        if not self.ai_enabled:
            response = "🤖 Test result feature unavailable. Please contact your health provider directly!"
            if self._is_swahili(user_message):
                response = "🤖 Kipengele cha matokeo ya mtihani hakipatikani. Tafadhali wasiliana na mtoa huduma wako wa afya moja kwa moja!"
            return response

        try:
            response = (
                "🤖 I couldn't find your test results. Please provide your AFYA KE ID or contact your health provider."
            )
            if self._is_swahili(user_message):
                response = (
                    "🤖 Sikuweza kupata matokeo yako ya mtihani. Tafadhali toa kitambulisho chako cha AFYA KE au wasiliana na mtoa huduma wako wa afya."
                )
            return response
        except Exception as e:
            logger.error(f"Error processing test result request for {phone_number}: {e}", exc_info=True)
            response = (
                "🤖 Unable to retrieve test results right now. Please contact your health provider directly!"
            )
            if self._is_swahili(user_message):
                response = (
                    "🤖 Sishindikani kupata matokeo ya mtihani sasa hivi. Tafadhali wasiliana na mtoa huduma wako wa afya moja kwa moja!"
                )
            return response

    def generate_medical_response(self, user_message: str, conversation_history: List[Dict] = None, phone_number: str = None, user_name: str = None, session_id: str = None) -> tuple[str, bool, str, str]:
        """
        Generates an AI response for gynecology-related inquiries using RAG.
        """
        if not user_message or not isinstance(user_message, str):
            logger.error(f"Invalid user message: {user_message}")
            response = "🤖 Sorry, I couldn't understand your request. Please try again."
            return response, False, None, None

        if not self.ai_enabled or not self.agent_executor:
            response = (
                "🤖 Sorry, I'm currently offline. Please contact a health provider directly!"
            )
            if self._is_swahili(user_message):
                response = (
                    "🤖 Samahani, niko nje ya mtandao sasa hivi. Tafadhali wasiliana na mtoa huduma wa afya moja kwa moja!"
                )
            return response, False, None, None

        # Handle kit requests
        if self._is_kit_request(user_message):
            return self.process_kit_request(user_message, phone_number, user_name, session_id)

        # Handle test result requests
        if self._is_result_request(user_message):
            response = self.process_result_request(user_message, phone_number, user_name, session_id)
            return response, False, None, None

        # Generate RAG-based response
        try:
            logger.info(f"Generating medical response for: {user_message[:100]}")
            
            # Prepare conversation history for context
            chat_history = []
            if conversation_history:
                for exchange in conversation_history[-5:]:
                    chat_history.append(("human", exchange.get("user", "")))
                    chat_history.append(("ai", exchange.get("assistant", "")))

            # Invoke RAG agent
            result = self.agent_executor.invoke({
                "input": user_message,
                "chat_history": chat_history
            })
            
            ai_response = result.get("output", "")
            
            if not ai_response or ai_response.strip() == "":
                raise ValueError("Empty response from agent")
            
            logger.info(f"Generated response: {ai_response[:100]}")

            # Don't add extra disclaimers if response already contains guidance
            if "contact" not in ai_response.lower() and "health" not in ai_response.lower():
                if self._is_swahili(user_message):
                    ai_response = (
                        f"{ai_response}\n\n💙 Kwa maelezo zaidi, wasiliana na mhudumu wa afya."
                    )
                else:
                    ai_response = (
                        f"{ai_response}\n\n💙 For personalized advice, please consult a healthcare professional."
                    )
            
            return ai_response, False, None, None

        except Exception as e:
            logger.error(f"Error generating medical response for message '{user_message}': {e}", exc_info=True)
            response = (
                "🤖 I'm having trouble processing your request right now. Please contact a health provider directly!"
            )
            if self._is_swahili(user_message):
                response = (
                    "🤖 Nina shida kushughulikia ombi lako sasa hivi. Tafadhali wasiliana na mtoa huduma wa afya moja kwa moja!"
                )
            return response, False, None, None