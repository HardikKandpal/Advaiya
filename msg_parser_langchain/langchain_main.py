import argparse
import asyncio
import os
import logging
from datetime import datetime
import uuid
import warnings
import sys
import torch

# Import the utility function first
from utils import suppress_cuda_warnings, get_system_metrics, log_performance_metrics

# Suppress CUDA warnings before any other imports
restore_stderr = suppress_cuda_warnings()

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to suppress TensorFlow logging more aggressively
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
except ImportError:
    pass

# Import your module
from langchain_extractor import LangchainPropertyExtractor

# Restore stderr after imports
restore_stderr()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('property_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

# Add timeout handling for model loading
import signal
from contextlib import contextmanager
import gc

class TimeoutException(Exception):
    pass

# Replace the time_limit implementation with a Windows-compatible version
import threading
import ctypes
VLLM_AVAILABLE = False
@contextmanager
def time_limit(seconds):
    """Windows-compatible timeout context manager"""
    timer = None
    
    def quit_function(tid):
        """Terminate the thread with the given ID"""
        tid = ctypes.c_long(tid)
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenThread(1, False, tid)
        if handle:
            kernel32.TerminateThread(handle, 0)
            kernel32.CloseHandle(handle)
    
    def interrupt_function():
        """Raise TimeoutException in the main thread"""
        thread_id = threading.get_ident()
        logger.warning(f"Time limit exceeded, interrupting thread {thread_id}")
        quit_function(thread_id)
        raise TimeoutException("Timed out!")
    
    try:
        timer = threading.Timer(seconds, interrupt_function)
        timer.start()
        yield
    finally:
        if timer:
            timer.cancel()

async def main():
    parser = argparse.ArgumentParser(description='Extract property information from chat files using Langchain')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--model', '-m', type=str, default="mistralai/Mistral-7B-Instruct-v0.3", 
                        help='Model path or identifier')
    parser.add_argument('--batch-size', '-b', type=int, default=2, help='Batch size for processing')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of messages to process')
    parser.add_argument('--sliding-window', '-s', type=int, default=300, 
                        help='Sliding window size for long texts')
    parser.add_argument('--quantize', '-q', type=str, choices=['4bit', '8bit', 'none'], default='4bit',
                        help='Quantization level for model loading')
    parser.add_argument('--device', '-d', type=str, default='auto',
                        help='Device mapping strategy (auto, cuda:0, etc.)')
    
    # Add new arguments for model loading
    parser.add_argument('--timeout', '-t', type=int, default=600,
                        help='Timeout in seconds for model loading')
    parser.add_argument('--smaller-model', '-sm', action='store_true',
                        help='Use a smaller model if the main model fails to load')
    parser.add_argument('--offload-folder', '-of', type=str, default='./offload',
                        help='Folder to use for offloading model weights')
    
    # Add new arguments for performance optimization
    parser.add_argument('--use-flash-attn', action='store_true',help='Use Flash Attention for faster inference')
    #parser.add_argument('--use-flash-attn', type=lambda x: x.lower() == 'true', default=False, help='Use Flash Attention (true/false)')
    parser.add_argument('--use-bettertransformer', action='store_true',
                        help='Use BetterTransformer for optimized inference')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for sampling')
    
    # Add a new argument for smaller models
    parser.add_argument('--alternative-model', type=str, 
                        help='Alternative model to try if main model fails')

    parser.add_argument('--no-use-vllm', action='store_true', 
                   help='Disable vLLM even if available')
    args = parser.parse_args()
    
    # Print GPU information with more details
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            print(f"CUDA device {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            # Add CUDA capability check
            major, minor = torch.cuda.get_device_capability(i)
            print(f"CUDA device {i} capability: {major}.{minor}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Generate output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_basename = os.path.basename(args.input)
        output_basename = f"{os.path.splitext(input_basename)[0]}_langchain_{timestamp}.json"
        args.output = os.path.join(os.path.dirname(args.input), output_basename)
    
    # Create offload folder if it doesn't exist
    if not os.path.exists(args.offload_folder):
        os.makedirs(args.offload_folder)
        print(f"Created offload folder: {args.offload_folder}")
    
    try:
        # Initialize the extractor with optimized settings
        print(f"Initializing LangchainPropertyExtractor with model {args.model}")
        print("DEBUG: About to start model loading")  # Add this line

        # Force garbage collection before loading model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache before model loading")

        # Try different models in sequence if needed
        models_to_try = [args.model]
        if args.alternative_model:
            models_to_try.append(args.alternative_model)
        if args.smaller_model:
            models_to_try.append("google/flan-t5-base")

        extractor = None
        for model_idx, model_path in enumerate(models_to_try):
            try:
                print(f"Attempting to load model {model_idx+1}/{len(models_to_try)}: {model_path}")
                print(f"DEBUG: Loading model {model_path}")  # Add this line
                
                # Log system metrics before model loading
                print("System metrics before model loading:")
                log_performance_metrics(get_system_metrics())
                
                # Try to load the model with a timeout
                try:
                    # Use a more robust approach for model loading
                    print(f"Loading model with timeout of {args.timeout} seconds")
                    
                    # Create a separate function for model loading to make timeout more reliable
                    def load_model():
                        return LangchainPropertyExtractor(
                            model_path=model_path,
                            batch_size=args.batch_size,
                            sliding_window_size=args.sliding_window,
                            quantization="none" if model_idx > 0 else args.quantize,  # Only use quantization for first model
                            device_map=args.device,
                            offload_folder=args.offload_folder,
                            use_flash_attn=args.use_flash_attn and model_idx == 0,  # Only use flash attention for first model
                            use_bettertransformer=args.use_bettertransformer and model_idx == 0,  # Only use bettertransformer for first model
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature
                        )
                    
                    #with time_limit(args.timeout):
                    extractor = load_model()
                    print("DEBUG: Model loaded successfully")  # Already present
                    
                    # ADD THIS LINE:
                    print("DEBUG: Model object created, about to log system metrics")
                
                    # If we get here, model loaded successfully
                    print(f"Successfully loaded model: {model_path}")
                    break
                    
                except TimeoutException:
                    logger.error(f"Model loading timed out after {args.timeout} seconds")
                    if model_idx == len(models_to_try) - 1:
                        raise TimeoutException(f"All models failed to load within timeout")
                    else:
                        print(f"Trying next model...")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error loading model {model_path}: {str(e)}", exc_info=True)
                    if model_idx == len(models_to_try) - 1:
                        raise
                    else:
                        print(f"Trying next model...")
                        continue
                
            except Exception as e:
                logger.error(f"Failed to initialize extractor with model {model_path}: {str(e)}", exc_info=True)
                if model_idx == len(models_to_try) - 1:
                    raise
        
        if extractor is None:
            raise RuntimeError("Failed to load any model")
        
        print("DEBUG: Passed extractor is not None check")  # ADD THIS LINE

        # Log system metrics after model loading
        print("System metrics after model loading:")
        # log_performance_metrics(get_system_metrics())  # Commented out for debugging
        print("DEBUG: Model loaded, about to process file")  # <--- Add this line
        
        # Process the file
        print(f"Processing file: {args.input}")
        try:
            start_time = datetime.now()
            print(f"Processing started at: {start_time}")
            
            # Add progress reporting
            print("Starting file processing...")
            
            try:
                # Add detailed timing for each step
                extraction_start = datetime.now()
                results = await extractor.process_file_async(
                    filepath=args.input,
                    output_path=args.output,
                    limit=args.limit
                )
                extraction_end = datetime.now()
                extraction_duration = (extraction_end - extraction_start).total_seconds()
            except Exception as e:
                logger.error(f"Error during extraction: {str(e)}", exc_info=True)
                print("Attempting to use a simpler extraction method...")
                
                # Fallback to a simpler extraction method
                from langchain_extractor import SimplePropertyExtractor
                simple_extractor = SimplePropertyExtractor()
                results = await simple_extractor.process_file_async(
                    filepath=args.input,
                    output_path=args.output,
                    limit=args.limit
                )
            
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            print(f"Processing complete. Results saved to {args.output}")
            print(f"Processed {len(results)} messages in {processing_duration:.2f} seconds")
            
            # Print summary statistics
            successful = sum(1 for r in results if r.get('properties') and len(r.get('properties', [])) > 0)
            print(f"Successfully extracted properties: {successful}/{len(results)} ({successful/len(results)*100:.2f}%)")
            
            # Print average processing time
            processing_times = [r.get('processing_time', 0) for r in results]
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            print(f"Average processing time per message: {avg_time:.2f}s")
            
            # Log final system metrics
            print("Final system metrics:")
            log_performance_metrics(get_system_metrics())
            
            return 0
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return 1
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    # Run the async main function
    try:
        print("DEBUG: Starting main logic")
        exit_code = asyncio.run(main())
    except Exception as e:
        print(f"UNCAUGHT EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        exit_code = 1
    
    exit(exit_code)