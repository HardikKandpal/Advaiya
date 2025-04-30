
VLLM_AVAILABLE = False
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import asyncio
import orjson
import time
from transformers import pipeline

import uuid
import logging
from tqdm import tqdm
import aiofiles
import os
from datetime import datetime
import nest_asyncio
import gc
import re
import outlines
logger = logging.getLogger()
log_dir = "/kaggle/working"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "debug_output.log")),
        logging.StreamHandler()
    ]
)
# Try to import vLLM - this will be our primary inference engine
#try:
    #from vllm import LLM, SamplingParams
    #VLLM_AVAILABLE = True
    #print("vLLM is available and will be used for faster inference")
#except ImportError:
    #VLLM_AVAILABLE = False
    #logger.warning("vLLM not available. Install with: pip install vllm")

# Try to import Outlines for structured generation
try:
    import outlines
    from outlines.models.transformers import Transformers
    from outlines.output import Json
    OUTLINES_AVAILABLE = True
    print("Outlines is available and will be used for structured generation")
except ImportError:
    OUTLINES_AVAILABLE = False
    logger.warning("Outlines not available. Install with: pip install outlines")

# Langchain imports
from langchain_community.llms import HuggingFacePipeline
try:
    from langchain_community.llms import VLLM as LangchainVLLM
    LANGCHAIN_VLLM_AVAILABLE = True
except ImportError:
    LANGCHAIN_VLLM_AVAILABLE = False
    logger.warning("LangChain VLLM integration not available")

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import your existing models and utility functions
from models import ResidentialProperty, CommercialProperty, LandProperty
from utils import get_system_metrics, log_performance_metrics

logger = logging.getLogger(__name__)

class LangchainPropertyExtractor:
    def __init__(self, model_path="mistralai/Mistral-7B-Instruct-v0.3", batch_size=2, 
                 sliding_window_size=256, cache_size=1000, quantization="none", 
                 device_map="auto", offload_folder="/kaggle/working/offload",
                 use_flash_attn=False, use_bettertransformer=False,
                 max_new_tokens=256, temperature=0.1):
        """Optimized initializer for T4 GPUs"""
        
        # Force-disable problematic features for T4
        VLLM_AVAILABLE = False
        use_bettertransformer = False
        use_flash_attn = False
        
        # Basic initialization
        self.batch_size = 2  # Fixed for T4 safety
        self.sliding_window_size = sliding_window_size
        self.metrics = {
            "total_messages": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "llm_call_count": 0,
            "llm_total_time": 0,
            "errors":{}
        }
        
        print(f"Loading model: {model_path}")
        start_time = time.time()
        
        try:
            # 1. Load Tokenizer (CPU first)
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                device_map="cpu"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # --- CHANGE THIS BLOCK ---
            print(f"Loading model with device_map={device_map}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            for name, param in self.model.named_parameters():
                print(name, param.device)
            # --- CHANGE THIS BLOCK ---
            print("Creating pipeline...")
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                batch_size=2
            )
            # --- END CHANGE ---

            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise

        # --- ADD THIS BLOCK: create HuggingFacePipeline and assign to self.langchain_llm ---
        from langchain_community.llms import HuggingFacePipeline
        self.langchain_llm = HuggingFacePipeline(pipeline=self.pipe)
        # --- END ADD ---

        self._create_property_chains()
        nest_asyncio.apply()
    
    def _start_monitoring(self):
        """Start a background task to monitor system resources"""
        # Initialize monitoring metrics
        self.monitoring_data = {
            "timestamps": [],
            "gpu_memory_used": [],
            "cpu_percent": [],
            "ram_percent": []
        }
        
        # Set monitoring flag
        self.monitoring_active = True
        
        # Start monitoring in background
        print("Starting system resource monitoring")
        
        # We'll just initialize the monitoring data structure
        # but not actually start a background thread for simplicity
        # In a production system, you might want to use a background thread
        
        # Log initial system metrics
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                mem_used = torch.cuda.memory_allocated(gpu_id) / 1024**2
                mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2
                print(f"GPU {gpu_id} memory: {mem_used:.1f}MB / {mem_total:.1f}MB ({mem_used/mem_total*100:.1f}%)")

    def _create_property_chains(self):
        """Create Langchain chains with improved prompts including few-shot examples"""
        extraction_instruction = """
        <s>[INST] You are a property information extraction expert.

        Extract all property listings from the following message. 
        A message may contain multiple properties, and different property types (residential, commercial, land) may appear together. 
        For each property, extract only the fields that are present in the message. 
        If a field is missing, skip it. 
        Do not copy from the examples. 
        Return a JSON array of objects, one for each property, preserving their individuality and type.

        Return only the JSON array, wrapped with <JSON> and </JSON> tags. Do not include any explanation or extra text.

        Examples:

        Message: 
        *DIRECT RENTAL 2BHK* UNFURNISHED 
        *WELL DONE UP FLAT*
        ▪️ *TULSI 1* ▪️ 7 BUNGALOWS
        🔸 *70K Rent / 2L Deposit*🔸
        BACHELOR'S ALLOWED
        Call for inspection Mayank 9833801940

        <JSON>
        [
        {{
            "property_category": "residential",
            "configuration": "2BHK",
            "location": "TULSI 1, 7 BUNGALOWS",
            "price": {{
                "rent": "70000",
                "deposit": "200000"
            }},
            "furnishing_status": "Unfurnished",
            "preferred_tenants": ["Bachelors"],
            "contact": ["9833801940"]
        }}
        ]
        </JSON>

        Message:
        Commercial Shop office ground floor Andheri East Subway ke pass chocolate ke pass
        Big Shop office space area 265 built- area
        One washroom hai 
        East West entry hai
        One Car Parking hai
        Price 1.00 cr Negotiable hai

        and 

        Commercial Shop 
        Ground floor Andheri East 
        Subway ke pass 
        Small Shop Office space area 141 built- up 
        One washroom hai 
        East west entry hai 
        One Car Parkings hai 
        Price 55 Lakh 
        Contact Shiv Narayan Gupta 9967171375

        <JSON>
        [
        {{
            "property_category": "commercial",
            "property_type": "shop",
            "location": "Andheri East Subway",
            "area": {{"value": "265", "unit": "built-up"}},
            "washroom": true,
            "parking_available": true,
            "price": {{"sale": "10000000"}},
            "additional_details": ["East West entry", "Negotiable"]
        }},
        {{
            "property_category": "commercial",
            "property_type": "shop",
            "location": "Andheri East Subway",
            "area": {{"value": "141", "unit": "built-up"}},
            "washroom": true,
            "parking_available": true,
            "price": {{"sale": "5500000"}},
            "contact": ["9967171375"],
            "additional_details": ["East West entry"]
        }}
        ]
        </JSON>

        Message:
        Confirm plot 60yards #ews flats #tdicity #ansal #parsvanath #omaxe etc #ahinsaproperties 9868355556

        <JSON>
        [
        {{
            "property_category": "land",
            "area": {{"value": "60", "unit": "yards"}},
            "contact": ["9868355556"],
            "additional_details": ["ews flats", "tdicity", "ansal", "parsvanath", "omaxe", "ahinsaproperties"]
        }}
        ]
        </JSON>

        Message:
        Direct Commercial & Residential on sale 
        1 ) jackpot Deal Available in Bandra West 
        Exclusive Prenium luxury independent Bunglow plot - 500 sq.mt on sale near Sachin Tendulkar Bunglow Bandra West price 75 cr negotiable 
        2) Commercial office on sale Hubtown Solaris Andheri East near Andheri station Oc received Office no 1 ) Carpet 3000 sqft  Office no 2 ) carpet 3500 sqft Office no 3 ) carpet 3750 sqft Office no 4 ) carpet 2444 sqft Price 35000 on carpet

        <JSON>
        [
        {{
            "property_category": "land",
            "plot_type": "bungalow plot",
            "location": "near Sachin Tendulkar Bunglow, Bandra West",
            "area": {{"value": "500", "unit": "sq.mt"}},
            "price": {{"sale": "750000000"}},
            "additional_details": ["Exclusive Premium luxury", "negotiable"]
        }},
        {{
            "property_category": "commercial",
            "property_type": "office",
            "location": "Hubtown Solaris, Andheri East near Andheri station",
            "area": {{"value": "3000", "unit": "sqft"}},
            "price": {{"sale": "105000000"}},
            "additional_details": ["Oc received"]
        }},
        {{
            "property_category": "commercial",
            "property_type": "office",
            "location": "Hubtown Solaris, Andheri East near Andheri station",
            "area": {{"value": "3500", "unit": "sqft"}},
            "price": {{"sale": "122500000"}},
            "additional_details": ["Oc received"]
        }},
        {{
            "property_category": "commercial",
            "property_type": "office",
            "location": "Hubtown Solaris, Andheri East near Andheri station",
            "area": {{"value": "3750", "unit": "sqft"}},
            "price": {{"sale": "131250000"}},
            "additional_details": ["Oc received"]
        }},
        {{
            "property_category": "commercial",
            "property_type": "office",
            "location": "Hubtown Solaris, Andheri East near Andheri station",
            "area": {{"value": "2444", "unit": "sqft"}},
            "price": {{"sale": "85540000"}},
            "additional_details": ["Oc received"]
        }}
        ]
        </JSON>

        MESSAGE: {text}

        [/INST]
        """

        # Use the same prompt for all property types, since mixed types are possible
        property_prompt = PromptTemplate(
            template=extraction_instruction,
            input_variables=["text"]
        )
        self.residential_chain = LLMChain(
            llm=self.langchain_llm, 
            prompt=property_prompt,
            verbose=False
        )
        self.commercial_chain = LLMChain(
            llm=self.langchain_llm, 
            prompt=property_prompt,
            verbose=False
        )
        self.land_chain = LLMChain(
            llm=self.langchain_llm, 
            prompt=property_prompt,
            verbose=False
        )

    def _extract_json_from_text(self, text, category=None):
        print(f"LLM raw output for parsing: {repr(text)[:500]}")  # Print
        # Extract all <JSON>...</JSON> blocks
        json_blocks = re.findall(r'<JSON>(.*?)</JSON>', text, re.DOTALL)
        if json_blocks:
            # Try parsing from the last block (most likely the answer)
            for json_str in reversed(json_blocks):
                json_str = json_str.strip()
                try:
                    print(f"Parsing JSON block: {json_str[:200]}")
                    result = orjson.loads(json_str)
                    print(f"Regex JSON result: {result}")
                    return result
                except Exception as e:
                    print(f"Failed to parse <JSON> block: {e}")
        # Fallback to first valid JSON-like block or array
        print("Trying fallback: first valid JSON-like block or array")
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                print(f"Parsing fallback JSON block: {json_str[:200]}")
                result = orjson.loads(json_str)
                print(f"Fallback JSON result: {result}")
                return result
            except Exception as e:
                print(f"Failed to parse fallback JSON block: {e}")
        print("No valid JSON found, returning None")
        return None

    def _normalize_property_data(self, property_data, category):
        if isinstance(property_data, str):
            property_data = self._extract_json_from_text(property_data, category)
        if isinstance(property_data, dict):
            property_data = [property_data]
        elif not isinstance(property_data, list):
            property_data = []
        for prop in property_data:
            if "property_category" not in prop:
                prop["property_category"] = category
        return property_data

        
    async def _read_file_async(self, filepath):
        """Read a file asynchronously in chunks"""
        print(f"DEBUG: _read_file_async called for {filepath}")
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                chunk_size = 8192  # 8KB chunks
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            raise

    async def _extract_messages_async(self, chunks):
        """Extract messages from text chunks asynchronously"""
        buffer = ""
        message_id = 0
        
        # Define message patterns
        message_start_pattern = re.compile(
            r'^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?:\s*[APMapm]{2})?\s*-\s+[^:]+:', re.MULTILINE
        )


        print("DEBUG: Starting message extraction...")  # DEBUG

        async for chunk in chunks:
            print(f"DEBUG: Read chunk of size {len(chunk)}")  # DEBUG
            buffer += chunk

            # Split buffer into lines for easier processing
            lines = buffer.splitlines()
            current_message = None
            current_lines = []

            for line in lines:
                print(f"DEBUG: Checking line: {repr(line)}") 
                if message_start_pattern.match(line):
                    print(f"DEBUG: Matched message start: {line}")  # DEBUG
                    # If we have a current message, yield it
                    if current_lines:
                        full_message = "\n".join(current_lines).strip()
                        print(f"DEBUG: Yielding message: {full_message[:100]}")  # DEBUG
                        # Extract timestamp, sender, and content
                        parts = full_message.split(" - ", 1)
                        if len(parts) == 2:
                            timestamp = parts[0].strip()
                            sender_content = parts[1].strip()
                            sender_parts = sender_content.split(":", 1)
                            if len(sender_parts) == 2:
                                sender = sender_parts[0].strip()
                                content = sender_parts[1].strip()
                                message = {
                                    "id": str(message_id),
                                    "timestamp": timestamp,
                                    "sender": sender,
                                    "text": content,
                                    "raw": full_message
                                }
                                message_id += 1
                                yield message
                    current_lines = []
                    # Start new message
                    current_lines = [line]
                else:
                    # Continuation of the current message
                    if current_lines is not None:
                        current_lines.append(line)

            # Keep only the last (possibly incomplete) message in buffer
            buffer = "\n".join(current_lines) if current_lines else ""

        # After all chunks, process any remaining message
        if buffer.strip():
            print(f"DEBUG: Processing remaining buffer: {buffer[:100]}")  # DEBUG
            full_message = buffer.strip()
            parts = full_message.split(" - ", 1)
            if len(parts) == 2:
                timestamp = parts[0].strip()
                sender_content = parts[1].strip()
                sender_parts = sender_content.split(":", 1)
                if len(sender_parts) == 2:
                    sender = sender_parts[0].strip()
                    content = sender_parts[1].strip()
                    message = {
                        "id": str(message_id),
                        "timestamp": timestamp,
                        "sender": sender,
                        "text": content,
                        "raw": full_message
                    }
                    print(f"DEBUG: Yielding last message: {full_message[:100]}")  # DEBUG
                    yield message
            
           
    async def _preprocess_item_async(self, item):
        """Preprocess a message item asynchronously"""
        # Create a unique ID if not present
        if "id" not in item:
            item["id"] = str(uuid.uuid4())[:8]
        
        # Store original text
        original_text = item["text"]
        
        # Preprocess text
        processed_text = self._preprocess_text(original_text)
        
        # Determine property category
        category = self._determine_property_category(processed_text)
        
        return {
            "id": item["id"],
            "text": processed_text,
            "original_text": original_text,
            "category": category
        }
    
    def _preprocess_text(self, text):
        """Preprocess text for better extraction"""
        # Convert to uppercase for better pattern matching
        text = text.upper()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _determine_property_category(self, text):
        """Determine property category from text"""
        text_lower = text.lower()
        
        # Check for commercial indicators
        if any(word in text_lower for word in ["office", "shop", "commercial", "retail", "showroom", "warehouse"]):
            return "commercial"
        
        # Check for land indicators
        if any(word in text_lower for word in ["plot", "land", "acre", "hectare", "ground"]) and not any(word in text_lower for word in ["flat", "apartment", "bhk"]):
            return "land"
        
        # Default to residential
        return "residential"
        
    async def _extract_property_with_vllm(self, text, category):
        extraction_id = str(uuid.uuid4())[:8]
        print(f"Starting vLLM extraction {extraction_id} for category {category}")

        processed_text = self._preprocess_text(text)

        if category == "residential":
            prompt_template = self.residential_chain.prompt.template
        elif category == "commercial":
            prompt_template = self.commercial_chain.prompt.template
        else:
            prompt_template = self.land_chain.prompt.template

        prompt = prompt_template.format(text=processed_text)

        start_time = time.time()
        self.metrics["llm_call_count"] += 1

        try:
            if OUTLINES_AVAILABLE and hasattr(self, "residential_generator"):
                if category == "residential":
                    property_data = await asyncio.to_thread(self.residential_generator, text)
                elif category == "commercial":
                    property_data = await asyncio.to_thread(self.commercial_generator, text)
                else:
                    property_data = await asyncio.to_thread(self.land_generator, text)
            else:
                outputs = await asyncio.to_thread(
                    self.vllm_model.generate,
                    prompt,
                    self.sampling_params
                )
                response = outputs[0].outputs[0].text
                logger.debug(f"Raw LLM response for extraction {extraction_id}:\n{response}")
                property_data = self._normalize_property_data(self._extract_json_from_text(response, category), category)

            if not property_data:
                logger.warning(f"Failed to extract JSON from response for {extraction_id}")
                try:
                    with open("failed_outputs.log", "a", encoding="utf-8") as f:
                        f.write(f"Extraction ID: {extraction_id}\nRaw Output:\n{response}\n\n")
                except Exception as log_e:
                    logger.warning(f"Failed to log raw output: {log_e}")
                if category == "residential":
                    property_data = [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]
                elif category == "commercial":
                    property_data = [CommercialProperty(property_type="unknown", intent="unknown").model_dump()]
                else:
                    property_data = [LandProperty(property_type="unknown", intent="unknown").model_dump()]

            for prop in property_data:
                if "property_category" not in prop:
                    prop["property_category"] = category

            llm_time = time.time() - start_time
            self.metrics["llm_total_time"] += llm_time
            self.metrics["successful_extractions"] += 1

            logger.debug(f"Extraction {extraction_id} completed in {llm_time:.2f}s")
            return property_data

        except Exception as e:
            error_type = type(e).__name__
            if error_type not in self.metrics["errors"]:
                self.metrics["errors"][error_type] = 0
            self.metrics["errors"][error_type] += 1

            logger.warning(f"Error in vLLM extraction {extraction_id}: {str(e)}")
            self.metrics["failed_extractions"] += 1

            if category == "residential":
                return [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]
            elif category == "commercial":
                return [CommercialProperty(property_type="unknown", intent="unknown").model_dump()]
            else:
                return [LandProperty(property_type="unknown", intent="unknown").model_dump()]

    async def _extract_property_with_langchain(self, text, category):
        print(f"DEBUG: _extract_property_with_langchain called for category {category}")
        extraction_id = str(uuid.uuid4())[:8]
        print(f"Starting Langchain extraction {extraction_id} for category {category}")

        processed_text = self._preprocess_text(text)

        if category == "residential":
            chain = self.residential_chain
        elif category == "commercial":
            chain = self.commercial_chain
        else:
            chain = self.land_chain

        start_time = time.time()
        self.metrics["llm_call_count"] += 1

        try:
            chain_response = await asyncio.to_thread(
                chain.invoke,
                {"text": processed_text}
            )
            print(f"Raw LangChain LLM response for extraction {extraction_id}:\n{chain_response}")

            # --- FIX: Always extract the 'text' field if chain_response is a dict ---
            if isinstance(chain_response, dict) and "text" in chain_response:
                chain_response_text = chain_response["text"]
            else:
                chain_response_text = chain_response

            print(f"Parsing chain_response_text for extraction {extraction_id}")
            property_data = self._normalize_property_data(self._extract_json_from_text(chain_response_text, category), category)
            print(f"Normalized property_data for extraction {extraction_id}: {property_data}")

            if not property_data:
                print(f"Failed to extract JSON from response for {extraction_id}")
                if category == "residential":
                    property_data = [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]
                elif category == "commercial":
                    property_data = [CommercialProperty(property_type="unknown", intent="unknown").model_dump()]
                else:
                    property_data = [LandProperty(property_type="unknown", intent="unknown").model_dump()]

            for prop in property_data:
                if "property_category" not in prop:
                    prop["property_category"] = category

            llm_time = time.time() - start_time
            self.metrics["llm_total_time"] += llm_time
            self.metrics["successful_extractions"] += 1

            print(f"Extraction {extraction_id} completed in {llm_time:.2f}s")
            return property_data

        except Exception as e:
            error_type = type(e).__name__
            if error_type not in self.metrics["errors"]:
                self.metrics["errors"][error_type] = 0
            self.metrics["errors"][error_type] += 1

            print(f"Error in Langchain extraction {extraction_id}: {str(e)}")
            self.metrics["failed_extractions"] += 1

            if category == "residential":
                return [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]
            elif category == "commercial":
                return [CommercialProperty(property_type="unknown", intent="unknown").model_dump()]
            else:
                return [LandProperty(property_type="unknown", intent="unknown").model_dump()]

    async def _process_batch_async(self, batch):
        print(f"DEBUG: _process_batch_async called with batch of size {len(batch)}")
        batch_id = str(uuid.uuid4())[:8]
        logger.debug(f"Starting async batch processing {batch_id} with {len(batch)} messages")

        preprocess_tasks = [self._preprocess_item_async(item) for item in batch]
        batch_items = await asyncio.gather(*preprocess_tasks)

        batch_results = []

        # --- BEGIN CHANGE: True batching with pipeline ---
        # Prepare prompts for the batch
        prompts = []
        ids = []
        categories = []
        original_texts = []
        for item in batch_items:
            if item['category'] == "residential":
                prompt_template = self.residential_chain.prompt.template
            elif item['category'] == "commercial":
                prompt_template = self.commercial_chain.prompt.template
            else:
                prompt_template = self.land_chain.prompt.template
            prompt = prompt_template.format(text=item['text'])
            prompts.append(prompt)
            ids.append(item['id'])
            categories.append(item['category'])
            original_texts.append(item['original_text'])

        # Run the pipeline in batch
        start_time = time.time()
        self.metrics["llm_call_count"] += 1
        # Use asyncio.to_thread to avoid blocking event loop
        responses = await asyncio.to_thread(self.pipe, prompts)
        batch_time = time.time() - start_time
        logger.debug(f"Batch inference completed in {batch_time:.2f}s")

        for i, response in enumerate(responses):
            item_id = ids[i]
            category = categories[i]
            original_text = original_texts[i]
            item_start_time = time.time()
            try:
                # --- BEGIN FIX ---
                # Always extract the generated text string for parsing
                if isinstance(response, dict) and "generated_text" in response:
                    response_text = response["generated_text"]
                elif isinstance(response, str):
                    response_text = response
                elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) and "generated_text" in response[0]:
                    # Handle case where response is a list of dicts
                    response_text = response[0]["generated_text"]
                else:
                    response_text = str(response)
                # --- END FIX ---
                property_data = self._normalize_property_data(self._extract_json_from_text(response_text, category), category)
                if not property_data:
                    if category == "residential":
                        property_data = [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]
                    elif category == "commercial":
                        property_data = [CommercialProperty(property_type="unknown", intent="unknown").model_dump()]
                    else:
                        property_data = [LandProperty(property_type="unknown", intent="unknown").model_dump()]
                for prop in property_data:
                    if "property_category" not in prop:
                        prop["property_category"] = category
                self.metrics["successful_extractions"] += 1
            except Exception as e:
                logger.warning(f"Error processing item {item_id}: {str(e)}")
                if category == "residential":
                    property_data = [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]
                elif category == "commercial":
                    property_data = [CommercialProperty(property_type="unknown", intent="unknown").model_dump()]
                else:
                    property_data = [LandProperty(property_type="unknown", intent="unknown").model_dump()]
                for prop in property_data:
                    prop["property_category"] = category
                self.metrics["failed_extractions"] += 1

            item_processing_time = time.time() - item_start_time
            result = {
                "id": item_id,
                "text": original_text,
                "properties": property_data,
                "processing_time": item_processing_time
            }
            batch_results.append(result)
        # --- END CHANGE ---

        return batch_results

    async def process_file_async(self, filepath, output_path, limit=None):
        """Process a chat file asynchronously with optimized batch processing"""
        print(f"DEBUG: Entered process_file_async for {filepath}")
        process_id = str(uuid.uuid4())[:8]
        print(f"Starting file processing {process_id} for {filepath}")
        start_time = time.time()
        
        try:
            print(f"Reading file: {filepath}")
            
            # Create an async generator for reading the file in chunks
            chunks = self._read_file_async(filepath)
            
            # Create an async generator for extracting messages from chunks
            messages_generator = self._extract_messages_async(chunks)
            
            # Apply limit if specified
            if limit:
                print(f"Processing limited to {limit} messages")
                limited_messages = []
                count = 0
                async for message in messages_generator:
                    limited_messages.append(message)
                    count += 1
                    if count >= limit:
                        break
                messages = limited_messages
            else:
                # Collect all messages
                messages = [message async for message in messages_generator]
            print(f"DEBUG: Total messages extracted: {len(messages)}")  # DEBUG
            
            self.metrics["total_messages"] += len(messages)
            print(f"Extracted {len(messages)} messages from {filepath}")
            
            # Process in batches for better GPU utilization
            all_results = []
            
            # Optimize batch size based on available technology
            if VLLM_AVAILABLE:
                effective_batch_size = self.batch_size
            else:
                #effective_batch_size = min(4, self.batch_size)
                effective_batch_size = self.batch_size
            batches = [messages[i:i + effective_batch_size] for i in range(0, len(messages), effective_batch_size)]
                
            
            # Use tqdm to show progress
            with tqdm(total=len(messages), desc="Processing messages") as pbar:
                for i, batch in enumerate(batches):
                    try:
                        # Process one batch at a time
                        batch_results = await self._process_batch_async(batch)
                        all_results.extend(batch_results)
                        pbar.update(len(batch))
                        
                        # Log progress more frequently
                        if i % 5 == 0 or i == len(batches) - 1:
                            print(f"Processed {len(all_results)}/{len(messages)} messages ({(len(all_results)/len(messages)*100):.1f}%)")
                            
                        # Calculate and log performance metrics
                        if batch_results:
                            batch_times = [r.get("processing_time", 0) for r in batch_results if "processing_time" in r]
                            if batch_times:
                                avg_time = sum(batch_times) / len(batch_times)
                                logger.debug(f"Batch {i+1}/{len(batches)}: avg processing time {avg_time*1000:.1f}ms per message")
                            
                    except Exception as e:
                        logger.error(f"Error processing batch {i}: {str(e)}")
                        # Create error results for this batch
                        for item in batch:
                            all_results.append({
                                "id": item['id'],
                                "text": item['text'],
                                "properties": [],
                                "error": f"Batch error: {str(e)}",
                                "processing_time": 0
                            })
                            pbar.update(1)
                    
                    # Clear CUDA cache after each batch to prevent memory buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    # Every few batches, do a more thorough cleanup
                    if i > 0 and i % 10 == 0:
                        print("Performing memory cleanup")
                        gc.collect()
                        if torch.cuda.is_available():
                            # Log GPU memory usage
                            for gpu_id in range(torch.cuda.device_count()):
                                mem_used = torch.cuda.memory_allocated(gpu_id) / 1024**2
                                mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2
                                print(f"GPU {gpu_id} memory: {mem_used:.1f}MB / {mem_total:.1f}MB ({mem_used/mem_total*100:.1f}%)")
                        
                        # Brief pause to let system stabilize
                        await asyncio.sleep(0.5)
            
            # Sort results by ID to maintain original order
            all_results.sort(key=lambda x: x["id"])
            
            # Calculate success rate and performance metrics
            success_count = sum(1 for r in all_results if r.get("properties") and len(r.get("properties", [])) > 0)
            success_rate = success_count / len(all_results) if all_results else 0
            
            processing_times = [r.get("processing_time", 0) for r in all_results if "processing_time" in r]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            print(f"Processing complete: {success_count}/{len(all_results)} successful extractions ({success_rate*100:.1f}%)")
            print(f"Average processing time: {avg_processing_time*1000:.1f}ms per message")
            
            # Save results
            if all_results:
                print(f"Saving {len(all_results)} results to {output_path}")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Use orjson for faster serialization
                with open(output_path, 'wb') as f:
                    f.write(orjson.dumps(
                        all_results, 
                        option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
                    ))
                print(f"Results saved successfully to {output_path}")
            else:
                logger.warning("No results to save")
            
            # Log final metrics
            total_time = time.time() - start_time
            final_metrics = {
                "file": filepath,
                "total_messages": len(messages),
                "total_results": len(all_results),
                "successful_extractions": success_count,
                "success_rate": success_rate,
                "total_time": total_time,
                "avg_time_per_message": total_time / len(messages) if messages else 0,
                "extraction_metrics": self.metrics
            }
            log_performance_metrics(final_metrics)
            print(f"File processing completed in {total_time:.2f}s")
            
            return all_results
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            # Log the error metrics
            error_metrics = {
                "file": filepath,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": time.time() - start_time
            }
            log_performance_metrics(error_metrics)
            raise