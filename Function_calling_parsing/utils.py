import psutil
import torch
import logging
import json
import os
import sys
import ctypes
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def suppress_cuda_warnings():
    """Suppress CUDA warnings using multiple methods"""
    # Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Try to redirect stderr at C level (Windows specific)
    try:
        if sys.platform == 'win32':
            # Get handle to kernel32.dll
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            
            # Constants for SetStdHandle
            STD_ERROR_HANDLE = -12
            
            # Get null device handle
            INVALID_HANDLE_VALUE = -1
            null_handle = kernel32.CreateFileW(
                'NUL', 
                0x40000000,  # GENERIC_WRITE
                0,  # No sharing
                None,  # No security attributes
                3,  # OPEN_EXISTING
                0,  # No flags
                None  # No template file
            )
            
            if null_handle != INVALID_HANDLE_VALUE:
                # Save original stderr handle
                original_stderr = kernel32.GetStdHandle(STD_ERROR_HANDLE)
                
                # Redirect stderr to null
                kernel32.SetStdHandle(STD_ERROR_HANDLE, null_handle)
                
                # Return a function to restore stderr
                def restore_stderr():
                    kernel32.SetStdHandle(STD_ERROR_HANDLE, original_stderr)
                    kernel32.CloseHandle(null_handle)
                
                return restore_stderr
    except Exception as e:
        logger.debug(f"Failed to redirect stderr at C level: {e}")
    
    # Fallback to Python-level redirection
    class StderrRedirector:
        def __init__(self):
            self.old_stderr = sys.stderr
            self.null_file = open(os.devnull, 'w')
        
        def start(self):
            sys.stderr = self.null_file
        
        def stop(self):
            sys.stderr = self.old_stderr
            self.null_file.close()
    
    redirector = StderrRedirector()
    redirector.start()
    
    return redirector.stop

def get_system_metrics():
    """Get system resource metrics"""
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3),
        "memory_used_gb": psutil.virtual_memory().used / (1024 ** 3),
        "disk_percent": psutil.disk_usage('/').percent,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add GPU metrics if available
    if torch.cuda.is_available():
        try:
            metrics["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                metrics[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
                metrics[f"gpu_{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024 ** 3)
                metrics[f"gpu_{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / (1024 ** 3)
                # Get memory stats if available
                try:
                    metrics[f"gpu_{i}_memory_total_gb"] = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                except:
                    pass
        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {str(e)}")
    
    return metrics

def log_performance_metrics(metrics):
    """Log performance metrics to file and console"""
    # Ensure metrics directory exists
    os.makedirs("metrics", exist_ok=True)
    
    # Log to file
    metrics_file = os.path.join("metrics", f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl")
    try:
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    except Exception as e:
        logger.warning(f"Error writing metrics to file: {str(e)}")
    
    # Log summary to console
    if "cpu_percent" in metrics:
        print(f"System: CPU {metrics['cpu_percent']}%, Memory {metrics['memory_percent']}%, "
                   f"Available {metrics['memory_available_gb']:.2f}GB")
    
    if "gpu_count" in metrics and metrics["gpu_count"] > 0:
        for i in range(metrics["gpu_count"]):
            if f"gpu_{i}_memory_allocated_gb" in metrics:
                print(f"GPU {i}: {metrics[f'gpu_{i}_name']} - "
                           f"Allocated {metrics[f'gpu_{i}_memory_allocated_gb']:.2f}GB")
    
    # Log application metrics if available
    if "extraction_metrics" in metrics:
        ext_metrics = metrics["extraction_metrics"]
        if "total_messages" in ext_metrics and ext_metrics["total_messages"] > 0:
            success_rate = ext_metrics["successful_extractions"] / ext_metrics["total_messages"] * 100
            print(f"Extraction: {ext_metrics['successful_extractions']}/{ext_metrics['total_messages']} "
                       f"successful ({success_rate:.2f}%)")