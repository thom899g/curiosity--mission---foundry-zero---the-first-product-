"""
Archon Core Processor - Three-Brain Architecture
Implements the specialized transformer for architectural analysis
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np

from config.settings import settings

@dataclass
class AnalysisResult:
    """Structured result from architectural analysis"""
    principles: List[Dict[str, Any]]
    prescriptions: List[str]
    confidence: float
    compute_tier: str
    analysis_id: str

class PrincipleExtractor:
    """Rule-based principle extraction from code patterns"""
    
    SECURITY_PRINCIPLES = {
        "authentication": ["auth", "login", "token", "jwt", "oauth"],
        "authorization": ["role", "permission", "access_control", "rbac"],
        "encryption": ["encrypt", "decrypt", "cipher", "aes", "rsa"],
        "input_validation": ["sanitize", "validate", "escape", "xss"],
        "logging": ["log", "audit", "trace", "monitor"]
    }
    
    PERFORMANCE_PRINCIPLES = {
        "caching": ["cache", "memcached", "redis", "ttl"],
        "compression": ["compress", "gzip", "deflate"],
        "connection_pooling": ["pool", "connection", "session"],
        "async_processing": ["async", "await", "coroutine", "future"]
    }
    
    def extract(self, code_blocks: List[str]) -> List[Dict[str, Any]]:
        """
        Extract architectural principles from code blocks
        Returns structured principles with confidence scores
        """
        principles = []
        
        for block in code_blocks:
            block_lower = block.lower()
            
            # Security principles
            for principle, keywords in self.SECURITY_PRINCIPLES.items():
                matches = [kw for kw in keywords if kw in block_lower]
                if matches:
                    confidence = len(matches) / len(keywords)
                    principles.append({
                        "type": "security",
                        "principle": principle,
                        "matches": matches,
                        "confidence": min(confidence * 1.5, 1.0),  # Cap at 1.0
                        "code_snippet": block[:100]  # First 100 chars
                    })
            
            # Performance principles
            for principle, keywords in self.PERFORMANCE_PRINCIPLES.items():
                matches = [kw for kw in keywords if kw in block_lower]
                if matches:
                    confidence = len(matches) / len(keywords)
                    principles.append({
                        "type": "performance",
                        "principle": principle,
                        "matches": matches,
                        "confidence": min(confidence * 1.5, 1.0),
                        "code_snippet": block[:100]
                    })
        
        # Deduplicate similar principles
        unique_principles = []
        seen = set()
        for principle in principles:
            key = f"{principle['type']}:{principle['principle']}:{principle['code_snippet']}"
            if key not in seen:
                seen.add(key)
                unique_principles.append(principle)
        
        logger.info(f"Extracted {len(unique_principles)} unique principles from {len(code_blocks)} code blocks")
        return unique_principles

class ContextEmbedder:
    """Vector embedding for code and business context"""
    
    def __init__(self):
        try:
            # Use a lightweight sentence transformer for embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384  # Dimension of the chosen model
            logger.success("ContextEmbedder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContextEmbedder: {e}")
            raise
    
    def embed(self, text_blocks: List[str], store_metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Generate embeddings for text blocks
        Returns normalized embeddings as numpy array
        """
        if not text_blocks:
            logger.warning("No text blocks provided for embedding")
            return np.array([])
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(text_blocks, convert_to_numpy=True)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_embeddings = embeddings / norms
            
            logger.debug(f"Generated {len(normalized_embeddings)} embeddings of dimension {self.embedding_dim}")
            return normalized_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(text_blocks), self.embedding_dim))

class ArchitectureAdvisor:
    """Fine-tuned transformer model for architectural prescriptions"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.model_name
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transformer model with error handling"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=settings.max_sequence_length // 2,
                temperature=0.7,
                do_sample=True
            )
            
            logger.success(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ArchitectureAdvisor: {e}")
            # Fallback to a simple rule-based advisor
            self.generator = None
    
    def generate(self, principles: List[Dict[str, Any]], 
                context_embeddings: np.ndarray,
                memory_context: Optional[List[str]] = None) -> List[str]:
        """
        Generate architectural prescriptions based on principles and context
        """
        if self.generator is None:
            return self._fallback_prescriptions(principles)
        
        try:
            # Prepare prompt from principles
            principle_text = "\n".join([
                f"- {p['type'].upper()}: {p['principle']} (confidence: {p['confidence']:.2f})"
                for p in principles[:5]  # Limit to top 5 principles
            ])
            
            prompt = f"""Analyze these architectural principles and provide specific, actionable prescriptions:

{principle_text}

Context: Analyzing code architecture for security and performance.

Prescriptions:
1."""
            
            # Generate response
            result = self.generator(prompt, max_length=settings.max_sequence_length)
            generated_text = result[0]['generated_text']
            
            # Extract prescriptions (lines starting with numbers)
            prescriptions = []
            lines = generated_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit() and '.' in line:
                    prescriptions.append(line)
            
            # Ensure we have at least 3 prescriptions
            if len(prescriptions) < 3:
                prescriptions.extend(self._fallback_prescriptions(principles)[:3-len(prescriptions)])
            
            logger.info(f"Generated {len(prescriptions)} architectural prescriptions")
            return prescriptions[:5]  # Return top 5 prescriptions
            
        except Exception as e:
            logger.error(f"Prescription generation failed: {e}")
            return self._fallback_prescriptions(principles)
    
    def _fallback_prescriptions(self, principles: List[Dict[str, Any]]) -> List[str]:
        """Fallback rule-based prescriptions when model fails"""
        prescriptions = []
        
        for principle in principles[:3]:
            principle_type = principle['type']
            principle_name = principle['principle']
            
            if principle_type == 'security':
                if principle_name == 'authentication':
                    prescriptions.append("1. Implement multi-factor authentication with time-based one-time passwords")
                elif principle_name == 'encryption':
                    prescriptions.append("2. Use AES-256-GCM for data at rest and TLS 1.3 for data in transit")
            elif principle_type == 'performance':
                if principle_name == 'caching':
                    prescriptions.append("3. Implement Redis with LRU eviction policy and 5-minute TTL for frequently accessed data")
        
        # Add generic prescriptions if needed
        while len(prescriptions) < 3:
            prescriptions.append(f"{len(prescriptions)+1}. Review and update dependency versions for security patches")
        
        return prescriptions

class ArchonProcessor:
    """Main processor coordinating the three-brain architecture"""
    
    def __init__(self):
        self.principle_brain = PrincipleExtractor()
        self.context_brain = ContextEmbedder()
        self.generative_brain = ArchitectureAdvisor()
        logger.info("ArchonProcessor initialized with three-brain architecture")
    
    def analyze(self, code_blocks: List[str], 
                business_context: Optional[str] = None,
                compute_tier: str = "tier1") -> AnalysisResult:
        """
        Main analysis method - coordinates all three brains
        Returns structured analysis result
        """
        import uuid
        analysis_id = str(uuid.uuid4())
        
        logger.info(f"Starting analysis {analysis_id} with {len(code_blocks)} code blocks")
        
        try:
            # Step 1: Extract principles
            principles = self.principle_brain.extract(code_blocks)
            
            # Step 2: Prepare context and generate embeddings
            context_items = code_blocks.copy()
            if business_context:
                context_items.append(f"Business Context: {business_context}")
            
            embeddings = self.context_brain.embed(context_items)
            
            # Step 3: Generate prescriptions
            prescriptions = self.generative_brain.generate(
                principles=principles,
                context_embeddings=embeddings
            )
            
            # Calculate overall confidence
            if principles:
                avg_confidence = sum(p['confidence'] for p in principles) / len(principles)
            else:
                avg_confidence = 0.3  # Low confidence if no principles detected
            
            result = AnalysisResult(
                principles=principles,
                prescriptions=prescriptions,
                confidence=round(avg_confidence, 2),
                compute_tier=compute_tier,
                analysis_id=analysis_id
            )
            
            logger.success(f"Analysis {analysis_id} completed with confidence {result.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {analysis_id}: {e}")
            # Return minimal result with error information
            return AnalysisResult(
                principles=[],
                prescriptions=[f"Analysis failed: {str(e)[:100]}"],
                confidence=0.0,
                compute_tier=compute_tier,
                analysis_id=analysis_id