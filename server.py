from fastapi import FastAPI, UploadFile, File ,HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from faster_whisper import WhisperModel, BatchedInferencePipeline
from contextlib import asynccontextmanager
from starlette.background import BackgroundTask
import uvicorn
import tempfile
import os
import traceback
import time
import gc
import asyncio
import json
import threading
import re

# GPU memory monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False
    print("⚠️  pynvml not available. Install with: pip install nvidia-ml-py3")
cuda_bin_path = fr"C:\Program Files\NVIDIA\CUDNN\v9.15\bin\12.9"

os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ["PATH"]

# Initialize model variables
model = None
batched_model = None

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if not NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "total_mb": info.total / (1024 ** 2),
            "used_mb": info.used / (1024 ** 2),
            "free_mb": info.free / (1024 ** 2),
            "used_percent": (info.used / info.total) * 100
        }
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None

def print_gpu_memory(label=""):
    """Print GPU memory usage with a label"""
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"💾 GPU Memory {label}: {mem_info['used_mb']:.1f}MB / {mem_info['total_mb']:.1f}MB ({mem_info['used_percent']:.1f}% used)")
    return mem_info

def classify_error(error: Exception) -> tuple[int, str]:
    """
    Classify error and return appropriate HTTP status code and error message.
    Returns: (status_code, error_message)
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Check for Out of Memory errors (OOM)
    oom_patterns = [
        r'out of memory',
        r'oom',
        r'cuda out of memory',
        r'cudaerroroutofmemory',
        r'insufficient memory',
        r'memory allocation failed',
        r'cannot allocate memory',
        r'ran out of memory'
    ]
    
    for pattern in oom_patterns:
        if re.search(pattern, error_str, re.IGNORECASE):
            return (500, f"Out of Memory Error: {str(error)}")
    
    # Check for CUDA/GPU related errors (server errors)
    cuda_patterns = [
        r'cuda',
        r'gpu',
        r'device',
        r'cudnn',
        r'nvidia'
    ]
    
    for pattern in cuda_patterns:
        if re.search(pattern, error_str, re.IGNORECASE):
            return (500, f"GPU/CUDA Error: {str(error)}")
    
    # Check for file format/validation errors (client errors)
    client_error_patterns = [
        r'invalid file',
        r'unsupported format',
        r'file format',
        r'cannot decode',
        r'corrupted',
        r'invalid audio',
        r'file too large',
        r'file size'
    ]
    
    for pattern in client_error_patterns:
        if re.search(pattern, error_str, re.IGNORECASE):
            return (400, f"Invalid Request: {str(error)}")
    
    # Check for file not found errors
    if 'file not found' in error_str or 'no such file' in error_str or 'filenotfound' in error_type.lower():
        return (400, f"File Error: {str(error)}")
    
    # Check for permission errors
    if 'permission' in error_str or 'permissiondenied' in error_type.lower():
        return (403, f"Permission Error: {str(error)}")
    
    # Check for timeout errors
    if 'timeout' in error_str or 'timed out' in error_str:
        return (504, f"Request Timeout: {str(error)}")
    
    # Default: treat as server error (500)
    return (500, f"Internal Server Error: {str(error)}")

def cleanup_gpu_memory():
    """Clean up GPU memory by deleting models and clearing caches"""
    global model, batched_model
    
    # Check if there's anything to clean up
    has_models = (model is not None) or (batched_model is not None)
    if not has_models:
        return  # Nothing to clean up, skip
    
    mem_before_cleanup = print_gpu_memory("(before cleanup)")
    print("🧹 Cleaning up GPU memory...")
    
    # Delete batched_model first
    if batched_model is not None:
        try:
            # Clear any internal references
            if hasattr(batched_model, 'model'):
                batched_model.model = None
            if hasattr(batched_model, 'last_speech_timestamp'):
                batched_model.last_speech_timestamp = 0.0
            del batched_model
            batched_model = None
        except Exception as e:
            print(f"Warning: Error deleting batched_model: {e}")
    
    # Delete model and its components
    if model is not None:
        try:
            # Try to access and delete CTranslate2 model directly
            if hasattr(model, 'model') and model.model is not None:
                try:
                    # Try to unload CTranslate2 model - this should free GPU memory
                    ctranslate2_model = model.model
                    # CTranslate2 models are stored in a Models object
                    # Try to delete the internal model storage
                    if hasattr(ctranslate2_model, '_model_storage'):
                        del ctranslate2_model._model_storage
                    # Delete the model object
                    del ctranslate2_model
                except Exception as e:
                    print(f"Warning: Error unloading CTranslate2 model: {e}")
                model.model = None
            
            # Delete feature extractor
            if hasattr(model, 'feature_extractor') and model.feature_extractor is not None:
                del model.feature_extractor
                model.feature_extractor = None
            
            # Delete decoder if exists
            if hasattr(model, 'decoder') and model.decoder is not None:
                del model.decoder
                model.decoder = None
            
            # Delete tokenizer if exists
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                del model.tokenizer
                model.tokenizer = None
            
            del model
            model = None
        except Exception as e:
            print(f"Warning: Error deleting model: {e}")
    
    # Clear PyTorch CUDA cache if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Error clearing PyTorch cache: {e}")
    
    # Try to clear CTranslate2 cache if available
    try:
        import ctranslate2
        # CTranslate2 might have internal cache cleanup
        # Force garbage collection to trigger CTranslate2 cleanup
    except:
        pass
    
    # Force garbage collection multiple times
    for _ in range(5):  # More aggressive cleanup
        gc.collect()
    
    # Wait longer for memory to be freed
    time.sleep(1.0)
    
    mem_after_cleanup = print_gpu_memory("(after cleanup)")
    if mem_before_cleanup and mem_after_cleanup:
        freed = mem_before_cleanup['used_mb'] - mem_after_cleanup['used_mb']
        if freed > 0:
            print(f"✅ Freed {freed:.1f}MB of GPU memory")
        else:
            print(f"⚠️  GPU memory not fully freed (may take time for CTranslate2 to release)")
    print("✅ GPU memory cleanup complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Load the model
    global model, batched_model
    
    # Check if model is already loaded (prevent double loading)
    if model is not None or batched_model is not None:
        print("⚠️  Model already loaded, skipping...")
    else:
        # Check current GPU memory - if it's unusually high, we might be reloading
        # and old models are still in memory. Try to clean up first.
        mem_check = get_gpu_memory_info()
        if mem_check and mem_check['used_mb'] > 1500:  # If more than 1.5GB used, likely a reload
            print(f"⚠️  High GPU memory usage detected ({mem_check['used_mb']:.1f}MB) - attempting cleanup...")
            # Try to clean up any orphaned models
            cleanup_gpu_memory()
            # Wait a bit more for CTranslate2 to release memory
            time.sleep(2.0)
            mem_after_wait = get_gpu_memory_info()
            if mem_after_wait:
                print(f"💾 GPU Memory after cleanup wait: {mem_after_wait['used_mb']:.1f}MB")

        # Load the model once into GPU memory
        print("🔄 Loading model...")
        mem_before = print_gpu_memory("(before model load)")
        model = WhisperModel("small", device="cuda", compute_type="int8_float32") # int8_float32 is the best performance model after quantization
        # Create batched inference pipeline for better memory management with large files
        batched_model = BatchedInferencePipeline(model=model)
        mem_after = print_gpu_memory("(after model load)")
        if mem_before and mem_after:
            model_memory = mem_after['used_mb'] - mem_before['used_mb']
            print(f"📊 Model uses approximately {model_memory:.1f}MB of GPU memory")
        print("✅ Model loaded and ready.")
    
    yield  # Server is running
    
    # Shutdown: Clean up GPU memory
    print("🛑 Server shutting down, cleaning up GPU memory...")
    cleanup_gpu_memory()

# Initialize FastAPI app with lifespan
app = FastAPI(title="Faster Whisper Server", version="1.0", lifespan=lifespan)

@app.get("/health")
async def health():
    """Health check endpoint"""
    gpu_mem = get_gpu_memory_info()
    return {
        "status": "healthy",
        "service": "Faster Whisper Server",
        "model_loaded": model is not None,
        "gpu_memory": gpu_mem
    }

@app.get("/gpu")
async def gpu_info():
    """Get detailed GPU memory information"""
    gpu_mem = get_gpu_memory_info()
    if not gpu_mem:
        return {"error": "GPU memory info not available. Install nvidia-ml-py3: pip install nvidia-ml-py3"}
    
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return {
                "gpu_name": name,
                "temperature_c": temp,
                "utilization_gpu_percent": util.gpu,
                "utilization_memory_percent": util.memory,
                "memory": gpu_mem
            }
        except Exception as e:
            return {"error": str(e), "memory": gpu_mem}
    return {"memory": gpu_mem}


async def transcribe_with_keepalive(tmp_path: str, audio_filename: str):
    """Generator that yields keep-alive messages during transcription"""
    # Ensure model is loaded - check before starting
    if model is None or batched_model is None:
        # Clean up temp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        # Yield error instead of raising (since we're in a generator)
        error_response = json.dumps({
            "__error__": True,
            "status_code": 503,
            "detail": "Model is not loaded yet. Please wait for server startup to complete."
        })
        yield error_response
        return
    
    transcription_result = None
    transcription_error = None
    transcription_error_status = None
    transcription_done = threading.Event()
    transcription_start_time = None
    
    def run_transcription():
        """Run transcription in a separate thread"""
        nonlocal transcription_result, transcription_error, transcription_error_status, transcription_start_time
        try:
            transcription_start_time = time.time()
            print(f"🎧 Transcribing: {audio_filename}")
            
            # Monitor GPU memory before processing
            mem_before = print_gpu_memory("(before transcription)")
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # 👇 Use batched inference for better memory management with large files
            # Enable word_timestamps to match OpenAI format
            try:
                segments, info = batched_model.transcribe(
                    tmp_path, 
                    batch_size=4,  # Reduced for 4GB GPU - increase if you have more VRAM
                    word_timestamps=True,
                    # chunk_length=60,  # Process in 30-second chunks to avoid memory errors
                )
                segments = list(segments)  # The transcription will actually run here.
            finally:
                # Clear GPU cache after processing to free memory
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                mem_after = print_gpu_memory("(after transcription)")
                
                # Show memory delta
                if mem_before and mem_after:
                    delta = mem_after['used_mb'] - mem_before['used_mb']
                    if delta > 0:
                        print(f"⚠️  Memory increased by {delta:.1f}MB during transcription")
                    elif delta < 0:
                        print(f"✅ Memory freed: {abs(delta):.1f}MB")
            
            # Record end time for performance tracking
            end_time = time.time()
            duration = round(end_time - start_time, 2)

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            print(f"⏱️ Transcription completed in {duration} seconds")

            # Build segments in OpenAI format
            segments_formatted = []
            words_list = []
            
            for segment in segments:
                # Format segment to match OpenAI's structure
                segment_dict = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    "tokens": segment.tokens,
                    "temperature": round(segment.temperature, 2) if segment.temperature is not None else 0.0,
                    "avg_logprob": round(segment.avg_logprob, 2),
                    "compression_ratio": round(segment.compression_ratio, 2),
                    "no_speech_prob": round(segment.no_speech_prob, 2)
                }
                segments_formatted.append(segment_dict)
                
                # Collect words with word-level timestamps (following faster-whisper pattern)
                if segment.words:
                    for word in segment.words:
                        words_list.append({
                            "word": word.word,
                            "start": round(word.start, 2),
                            "end": round(word.end, 2)
                        })
                        # Debug: print word-level timestamps (can be removed in production)
                        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
            
            # Build full text
            full_text = " ".join([s.text.strip() for s in segments])
            
            # Format response to match OpenAI Whisper API format
            response = {
                "text": full_text.strip(),
                "task": "transcribe",
                "language": info.language,
                "duration": round(info.duration, 2),
                "segments": segments_formatted
            }
            
            # Add words array if available (OpenAI format includes this)
            if words_list:
                response["words"] = words_list
            
            print(f"🗣️ Detected {info.language} ({info.language_probability:.2f})")
            transcription_result = response
        except Exception as e:
            traceback.print_exc()
            # Classify error and get appropriate status code
            status_code, error_message = classify_error(e)
            transcription_error = error_message
            transcription_error_status = status_code
            print(f"❌ Transcription error [{status_code}]: {error_message}")
        finally:
            transcription_done.set()
    
    # Start transcription in a background thread
    transcription_thread = threading.Thread(target=run_transcription, daemon=True)
    transcription_thread.start()
    
    # Send keep-alive messages every 10 seconds while transcription is running
    # Use minimal data (just newline) to keep connection alive without polluting response
    keepalive_interval = 10  # seconds
    last_keepalive = None
    
    while not transcription_done.is_set():
        await asyncio.sleep(1)  # Check every second
        
        # Wait for transcription to actually start
        if transcription_start_time is None:
            continue
        
        # Send minimal keep-alive (whitespace) every 10 seconds to prevent timeout
        # Whitespace is ignored by JSON parsers, so it won't affect the final response
        current_time = time.time()
        if last_keepalive is None or (current_time - last_keepalive) >= keepalive_interval:
            # Send whitespace to keep connection alive (JSON parsers ignore leading whitespace)
            yield " "
            last_keepalive = current_time
    
    # Wait for thread to complete
    transcription_thread.join()
    
    # Clean up temp file
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception as e:
        print(f"Warning: Could not delete temp file {tmp_path}: {e}")
    
    # Handle errors by yielding error info (will be handled by endpoint)
    if transcription_error:
        # Yield error info with status code - endpoint will handle this
        status_code = transcription_error_status if transcription_error_status else 500
        error_response = json.dumps({
            "__error__": True,
            "status_code": status_code,
            "detail": transcription_error
        })
        yield error_response
        return
    
    # Send final clean result - this is the only actual JSON in the response
    final_response = json.dumps(transcription_result)
    yield final_response


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Ensure model is loaded
    if model is None or batched_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait for server startup to complete.")
    
    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Save the uploaded file temporarily
    tmp_path = None
    try:
        suffix = os.path.splitext(audio.filename)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        # Create a custom response that can handle errors with proper status codes
        # We'll check the first chunk for errors and return appropriate response
        class ErrorAwareStreamingResponse(Response):
            def __init__(self, generator, tmp_file_path, *args, **kwargs):
                self.generator = generator
                self.tmp_file_path = tmp_file_path
                self.error_checked = False
                super().__init__(*args, **kwargs)
            
            async def __call__(self, scope, receive, send):
                # Check first chunk for errors before streaming
                gen = self.generator
                first_chunk = None
                
                try:
                    # Get first chunk
                    try:
                        first_chunk = await gen.__anext__()
                    except StopAsyncIteration:
                        # Generator ended immediately - error
                        if self.tmp_file_path and os.path.exists(self.tmp_file_path):
                            try:
                                os.remove(self.tmp_file_path)
                            except:
                                pass
                        await send({
                            'type': 'http.response.start',
                            'status': 500,
                            'headers': [[b'content-type', b'application/json']],
                        })
                        await send({
                            'type': 'http.response.body',
                            'body': json.dumps({"error": "Transcription failed unexpectedly"}).encode(),
                        })
                        return
                    
                    # Check if first chunk is an error
                    if first_chunk and first_chunk.strip().startswith('{'):
                        try:
                            first_data = json.loads(first_chunk.strip())
                            if isinstance(first_data, dict) and first_data.get("__error__"):
                                # Error detected - return proper status code
                                error_status = first_data.get("status_code", 500)
                                error_detail = first_data.get("detail", "Unknown error")
                                # Clean up temp file
                                if self.tmp_file_path and os.path.exists(self.tmp_file_path):
                                    try:
                                        os.remove(self.tmp_file_path)
                                    except:
                                        pass
                                # Return JSONResponse with proper status
                                await send({
                                    'type': 'http.response.start',
                                    'status': error_status,
                                    'headers': [[b'content-type', b'application/json']],
                                })
                                await send({
                                    'type': 'http.response.body',
                                    'body': json.dumps({"error": error_detail}).encode(),
                                })
                                return
                        except json.JSONDecodeError:
                            pass  # Not JSON error, continue streaming
                    
                    # No error, start streaming response
                    await send({
                        'type': 'http.response.start',
                        'status': 200,
                        'headers': [[b'content-type', b'application/json']],
                    })
                    
                    # Yield first chunk
                    if first_chunk:
                        await send({
                            'type': 'http.response.body',
                            'body': first_chunk.encode() if isinstance(first_chunk, str) else first_chunk,
                            'more_body': True,
                        })
                    
                    # Stream remaining chunks
                    try:
                        async for chunk in gen:
                            await send({
                                'type': 'http.response.body',
                                'body': chunk.encode() if isinstance(chunk, str) else chunk,
                                'more_body': True,
                            })
                    except StopAsyncIteration:
                        pass
                    
                    # Final chunk
                    await send({
                        'type': 'http.response.body',
                        'body': b'',
                        'more_body': False,
                    })
                    
                except Exception as e:
                    traceback.print_exc()
                    # Clean up temp file
                    if self.tmp_file_path and os.path.exists(self.tmp_file_path):
                        try:
                            os.remove(self.tmp_file_path)
                        except:
                            pass
                    # Classify error
                    status_code, error_message = classify_error(e)
                    # Send error response
                    await send({
                        'type': 'http.response.start',
                        'status': status_code,
                        'headers': [[b'content-type', b'application/json']],
                    })
                    await send({
                        'type': 'http.response.body',
                        'body': json.dumps({"error": error_message}).encode(),
                    })
        
        # Use custom response that handles errors with proper status codes
        return ErrorAwareStreamingResponse(
            transcribe_with_keepalive(tmp_path, audio.filename),
            tmp_path
        )
        
    except HTTPException:
        # Re-raise HTTPException (already has correct status code)
        raise
    except Exception as e:
        traceback.print_exc()
        # Clean up temp file on error
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        # Classify error and return appropriate status code
        status_code, error_message = classify_error(e)
        raise HTTPException(status_code=status_code, detail=error_message)


if __name__ == "__main__":
    # reload=True enables auto-reload on code changes (development mode)
    # Using import string format "server:app" is required for reload to work
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
