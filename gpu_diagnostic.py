"""
GPU Diagnostic Script
Run this to check if your GPU is properly configured for PyTorch
"""

import sys

print("="*70)
print("GPU DIAGNOSTIC TOOL")
print("="*70)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check PyTorch
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch installed: {torch.__version__}")
except ImportError:
    print("   ❌ PyTorch not installed!")
    print("   Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# Check CUDA availability
print("\n3. CUDA Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA Version (PyTorch): {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"      Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"      Multi-Processors: {props.multi_processor_count}")
else:
    print("   ❌ CUDA NOT AVAILABLE")
    print("\n   Possible reasons:")
    print("   1. PyTorch installed without CUDA support (CPU-only version)")
    print("   2. NVIDIA GPU drivers not installed or outdated")
    print("   3. No compatible NVIDIA GPU in system")
    print("\n   To fix:")
    print("   - Check if you have an NVIDIA GPU")
    print("   - Update NVIDIA drivers from https://www.nvidia.com/download/index.aspx")
    print("   - Reinstall PyTorch with CUDA:")
    print("     pip uninstall torch torchvision torchaudio")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Test GPU computation
if torch.cuda.is_available():
    print("\n4. Testing GPU Computation...")
    try:
        # Create a tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Perform computation
        z = torch.matmul(x, y)
        
        print(f"   ✅ GPU computation successful!")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")

# Check if system can see NVIDIA GPU
print("\n5. System GPU Check:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✅ nvidia-smi output:")
        print(result.stdout)
    else:
        print("   ❌ nvidia-smi not found or failed")
        print("   This means NVIDIA drivers are not properly installed")
except FileNotFoundError:
    print("   ❌ nvidia-smi command not found")
    print("   NVIDIA drivers may not be installed")

# Check diffusers
print("\n6. Checking Diffusers library...")
try:
    import diffusers
    print(f"   ✅ Diffusers installed: {diffusers.__version__}")
except ImportError:
    print("   ❌ Diffusers not installed")
    print("   Install with: pip install diffusers")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

if torch.cuda.is_available():
    print("\n✅ Your GPU is properly configured and ready to use!")
else:
    print("\n⚠️  GPU not available. Please follow the fix suggestions above.")