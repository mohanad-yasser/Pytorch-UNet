import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory Allocated:", torch.cuda.memory_allocated(0), "bytes")
    print("GPU Memory Reserved:", torch.cuda.memory_reserved(0), "bytes")
else:
    print("Using CPU only")
