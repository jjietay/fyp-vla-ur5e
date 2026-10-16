# Speech is local faster-whisper, push to talk, float16

Local rather than cloud because the examiner demo must not depend on the campus network; cost is under a dollar either way. Push to talk (Enter to start, Enter to stop) removes endpointing and works over SSH. `compute_type` must be float16, because int8 crashes on sm_120, and cuBLAS 12 and cuDNN 9 are preloaded in `shared/speech.py` because nothing else in this torch-free venv pulls them in. Both architectures receive the same raw transcript; cleaning it for B would put an LLM inside B.
