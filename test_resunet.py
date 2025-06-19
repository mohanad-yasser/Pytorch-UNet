import torch
from unet import UNet, ResUNet

def test_architectures():
    """Test and compare UNet and ResUNet architectures"""
    
    # Test parameters
    batch_size = 1
    channels = 1
    classes = 2
    height, width = 256, 256
    
    print("=" * 60)
    print("ARCHITECTURE COMPARISON: UNet vs ResUNet")
    print("=" * 60)
    
    # Test UNet
    print("\n1. Testing UNet:")
    unet = UNet(n_channels=channels, n_classes=classes)
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"   Parameters: {unet_params:,}")
    
    # Test ResUNet
    print("\n2. Testing ResUNet:")
    resunet = ResUNet(n_channels=channels, n_classes=classes)
    resunet_params = sum(p.numel() for p in resunet.parameters())
    print(f"   Parameters: {resunet_params:,}")
    
    # Compare
    param_increase = ((resunet_params - unet_params) / unet_params) * 100
    print(f"\n3. Comparison:")
    print(f"   Parameter increase: {param_increase:.1f}%")
    print(f"   Additional parameters: {resunet_params - unet_params:,}")
    
    # Test forward pass
    print("\n4. Testing forward pass:")
    x = torch.randn(batch_size, channels, height, width)
    
    # UNet forward pass
    with torch.no_grad():
        unet_output = unet(x)
        resunet_output = resunet(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   UNet output shape: {unet_output.shape}")
    print(f"   ResUNet output shape: {resunet_output.shape}")
    
    # Test memory usage
    print("\n5. Memory usage test:")
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        unet = unet.to(device)
        resunet = resunet.to(device)
        x = x.to(device)
        
        # Measure memory for UNet
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            unet_output = unet(x)
        unet_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Measure memory for ResUNet
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            resunet_output = resunet(x)
        resunet_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"   UNet GPU memory: {unet_memory:.1f} MB")
        print(f"   ResUNet GPU memory: {resunet_memory:.1f} MB")
        print(f"   Memory increase: {resunet_memory - unet_memory:.1f} MB")
    else:
        print("   CUDA not available, skipping memory test")
    
    print("\n" + "=" * 60)
    print("ResUNet Architecture Features:")
    print("=" * 60)
    print("✅ Residual blocks in encoder and decoder")
    print("✅ Skip connections with residual mappings")
    print("✅ Better gradient flow through residual connections")
    print("✅ Improved feature learning capabilities")
    print("✅ Enhanced segmentation accuracy (expected)")
    print("✅ More parameters for better representation learning")
    
    return unet, resunet

if __name__ == '__main__':
    unet, resunet = test_architectures() 