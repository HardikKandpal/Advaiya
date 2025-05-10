import torch
import asyncio
import orjson
import time
import uuid
import logging
from tqdm import tqdm
import aiofiles
import os
from datetime import datetime
import nest_asyncio
import gc
import re

from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from models import ResidentialProperty, CommercialProperty, LandProperty
from utils import get_system_metrics, log_performance_metrics

logger = logging.getLogger(__name__)

class PropertyExtractorFunctionCalling:
    def __init__(self, tokenizer_path, model_path, batch_size=2, max_new_tokens=256):
        self.batch_size = batch_size
        self.metrics = {
            "total_messages": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "llm_call_count": 0,
            "llm_total_time": 0,
            "errors":{}
        }
        print("Loading Mistral tokenizer and model for function calling...")
        self.tokenizer = MistralTokenizer.from_file(tokenizer_path)
        self.model = Transformer.from_folder(model_path)
        self.model = self.model.half()
        self.max_new_tokens = max_new_tokens

        # Define the property extraction function schema
        self.property_extraction_function = Function(
            name="extract_property_info",
            description="Extract property details from a message, and specify if it is a listing or an enquiry.",
            parameters={
                "type": "object",
                "properties": {
                    "property_category": {"type": "string"},
                    "property_type": {"type": "string"},
                    "intent": {
                        "type": "string",
                        "description": "Is this a property listing (offered) or an enquiry (wanted)? Use 'listed' or 'enquiry'."
                    },
                    "location": {"type": "string"},
                    "area": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "unit": {"type": "string"}
                        }
                    },
                    "price": {"type": "object"},
                    "contact": {"type": "array", "items": {"type": "string"}},
                    "additional_details": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["property_category", "intent"]
            }
        )
        self.property_tool = Tool(function=self.property_extraction_function)
        nest_asyncio.apply()

    async def _read_file_async(self, filepath):
        print(f"DEBUG: _read_file_async called for {filepath}")
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                chunk_size = 8192
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            raise

    async def _extract_messages_async(self, chunks):
        buffer = ""
        message_id = 0
        message_start_pattern = re.compile(
            r'^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?:\s*[APMapm]{2})?\s*-\s+[^:]+:', re.MULTILINE
        )
        print("DEBUG: Starting message extraction...")

        async for chunk in chunks:
            buffer += chunk
            lines = buffer.splitlines()
            current_lines = []
            for line in lines:
                if message_start_pattern.match(line):
                    if current_lines:
                        full_message = "\n".join(current_lines).strip()
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
                    current_lines = [line]
                else:
                    if current_lines is not None:
                        current_lines.append(line)
            buffer = "\n".join(current_lines) if current_lines else ""
        if buffer.strip():
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
                    yield message

    async def _preprocess_item_async(self, item):
        if "id" not in item:
            item["id"] = str(uuid.uuid4())[:8]
        original_text = item["text"]
        processed_text = self._preprocess_text(original_text)
        category = self._determine_property_category(processed_text)
        return {
            "id": item["id"],
            "text": processed_text,
            "original_text": original_text,
            "category": category
        }

    def _preprocess_text(self, text):
        text = text.upper()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _determine_property_category(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ["office", "shop", "commercial", "retail", "showroom", "warehouse"]):
            return "commercial"
        if any(word in text_lower for word in ["plot", "land", "acre", "hectare", "ground"]) and not any(word in text_lower for word in ["flat", "apartment", "bhk"]):
            return "land"
        return "residential"

    def _determine_intent(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ["requirement", "looking for", "needed", "need", "want", "enquiry", "searching for"]):
            return "enquiry"
        return "listed"

    async def _extract_property_with_function_calling(self, text):
        extraction_id = str(uuid.uuid4())[:8]
        print(f"Starting function calling extraction {extraction_id}")
        processed_text = self._preprocess_text(text)
        system_instruction = (
            "Extract all properties from the following message. "
            "For each property, output a separate object with these fields: "
            "property_category (residential, commercial, land), property_type, intent (listed or enquiry), "
            "location, area, price, contact, additional_details. "
            "If there are multiple properties, output a list of property objects. "
            "Output valid JSON only."
        )
        completion_request = ChatCompletionRequest(
            tools=[self.property_tool],
            messages=[
                UserMessage(content=system_instruction + "\n" + processed_text)
            ]
        )
        start_time = time.time()
        self.metrics["llm_call_count"] += 1
        try:
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            out_tokens, _ = generate(
                [tokens], self.model, max_tokens=self.max_new_tokens, temperature=0.0,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
            result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            print("RAW MODEL OUTPUT:", result)
            # --- BEGIN PATCH: Robust JSON extraction using regex ---
            import re
            property_data = []
            try:
                # Try to extract a JSON array first
                match = re.search(r'(\[\s*{.*?}\s*\])', result, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    property_data = orjson.loads(json_str)
                else:
                    # Fallback: extract a single JSON object
                    match = re.search(r'({.*})', result, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        property_data = orjson.loads(json_str)
                        if isinstance(property_data, dict):
                            property_data = [property_data]
                # Always flatten to a list of property objects
                if isinstance(property_data, dict) and "properties" in property_data and isinstance(property_data["properties"], list):
                    property_data = property_data["properties"]
                elif isinstance(property_data, dict):
                    property_data = [property_data]
                elif not isinstance(property_data, list):
                    property_data = []
            except Exception as e:
                logger.warning(f"Failed to parse function calling output for {extraction_id}: {e}")
                property_data = []
            # --- END PATCH ---
            if not property_data:
                property_data = [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]
            for prop in property_data:
                if "property_category" not in prop or not prop["property_category"]:
                    logger.warning(f"Model did not return property_category for extraction {extraction_id}")
                    prop["property_category"] = self._determine_property_category(text)
                if "property_type" not in prop or not prop["property_type"]:
                    logger.warning(f"Model did not return property_type for extraction {extraction_id}")
                    prop["property_type"] = "unknown"
                if "intent" not in prop or not prop["intent"]:
                    prop["intent"] = self._determine_intent(text)
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
            self.metrics["failed_extractions"] += 1
            logger.warning(f"Error in function calling extraction {extraction_id}: {str(e)}")
            return [ResidentialProperty(property_type="unknown", intent="unknown").model_dump()]

    async def _process_batch_async(self, batch):
        print(f"DEBUG: _process_batch_async called with batch of size {len(batch)}")
        preprocess_tasks = [self._preprocess_item_async(item) for item in batch]
        batch_items = await asyncio.gather(*preprocess_tasks)
        batch_results = []
        start_time = time.time()
        self.metrics["llm_call_count"] += 1
        extraction_tasks = [self._extract_property_with_function_calling(item['text']) for item in batch_items]
        extraction_results = await asyncio.gather(*extraction_tasks)
        for i, property_data in enumerate(extraction_results):
            item = batch_items[i]
            item_processing_time = time.time() - start_time

            # --- BEGIN: New schema-aligned output construction ---
            # Extract basic info (id, timestamp, sender, etc.)
            # You can extend this extraction as you add more fields to your models or parsing logic
            basic_message_info = {
                "id": item.get("id"),
                # Example: extract date/time/source_number/file_name from item or item["raw"] if available
                "date": None,
                "time": None,
                "source_number": None,
                "file/chat_name": None
            }
            contact_info = {
                "contact_name": None,
                "contact_number": None,
                "agency_name": None
            }
            # Build structured_message_content with property listings
            structured_message_content = {}
            for idx, prop in enumerate(property_data):
                listing_key = f"property_listing_{chr(ord('A') + idx)}"
                structured_message_content[listing_key] = prop

            result = {
                "basic_message_info": {k: v for k, v in basic_message_info.items() if v is not None},
                "contact_info": {k: v for k, v in contact_info.items() if v is not None},
                "structured_message_content": structured_message_content,
                "processing_time": item_processing_time
            }
            batch_results.append(result)
            # --- END: New schema-aligned output construction ---
        return batch_results

    async def process_file_async(self, filepath, output_path, limit=8):
        print(f"DEBUG: Entered process_file_async for {filepath}")
        process_id = str(uuid.uuid4())[:8]
        print(f"Starting file processing {process_id} for {filepath}")
        start_time = time.time()
        try:
            print(f"Reading file: {filepath}")
            chunks = self._read_file_async(filepath)
            messages_generator = self._extract_messages_async(chunks)
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
                messages = [message async for message in messages_generator]
            print(f"DEBUG: Total messages extracted: {len(messages)}")
            self.metrics["total_messages"] += len(messages)
            print(f"Extracted {len(messages)} messages from {filepath}")
            all_results = []
            batches = [messages[i:i + self.batch_size] for i in range(0, len(messages), self.batch_size)]
            with tqdm(total=len(messages), desc="Processing messages") as pbar:
                for i, batch in enumerate(batches):
                    try:
                        batch_results = await self._process_batch_async(batch)
                        all_results.extend(batch_results)
                        pbar.update(len(batch))
                        if i % 5 == 0 or i == len(batches) - 1:
                            print(f"Processed {len(all_results)}/{len(messages)} messages ({(len(all_results)/len(messages)*100):.1f}%)")
                        if batch_results:
                            batch_times = [r.get("processing_time", 0) for r in batch_results if "processing_time" in r]
                            if batch_times:
                                avg_time = sum(batch_times) / len(batch_times)
                                logger.debug(f"Batch {i+1}/{len(batches)}: avg processing time {avg_time*1000:.1f}ms per message")
                    except Exception as e:
                        logger.error(f"Error processing batch {i}: {str(e)}")
                        for item in batch:
                            all_results.append({
                                "id": item.get('id', 'unknown'),  # Ensure 'id' is always present
                                "text": item.get('text', ''),
                                "properties": [],
                                "error": f"Batch error: {str(e)}",
                                "processing_time": 0
                            })
                            pbar.update(1)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if i > 0 and i % 10 == 0:
                        print("Performing memory cleanup")
                        gc.collect()
                        if torch.cuda.is_available():
                            for gpu_id in range(torch.cuda.device_count()):
                                mem_used = torch.cuda.memory_allocated(gpu_id) / 1024**2
                                mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2
                                print(f"GPU {gpu_id} memory: {mem_used:.1f}MB / {mem_total:.1f}MB ({mem_used/mem_total*100:.1f}%)")
                        await asyncio.sleep(0.5)
            all_results.sort(key=lambda x: x.get("id", "unknown"))
            success_count = sum(1 for r in all_results if r.get("properties") and len(r.get("properties", [])) > 0)
            success_rate = success_count / len(all_results) if all_results else 0
            processing_times = [r.get("processing_time", 0) for r in all_results if "processing_time" in r]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            print(f"Processing complete: {success_count}/{len(all_results)} successful extractions ({success_rate*100:.1f}%)")
            print(f"Average processing time: {avg_processing_time*1000:.1f}ms per message")
            if all_results:
                print(f"Saving {len(all_results)} results to {output_path}")
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(orjson.dumps(
                        all_results, 
                        option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
                    ))
                print(f"Results saved successfully to {output_path}")
            else:
                logger.warning("No results to save")
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
            error_metrics = {
                "file": filepath,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": time.time() - start_time
            }
            log_performance_metrics(error_metrics)
            raise