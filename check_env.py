# Save this as check_env.py and run: python check_env.py
import torch
import torch_geometric
import transformers
import sklearn

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Graph Geometric version: {torch_geometric.__version__}")
print(f"Transformers version: {transformers.__version__}")
print("--- All systems go! ---")