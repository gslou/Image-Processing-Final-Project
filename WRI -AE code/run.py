
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from autoencoder import PositionalAutoencoder, LPIPSWithL2Loss
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim, ms_ssim
from skimage.io import imsave
import openslide


def ssim_loss(output, target):
    return 1 - ssim(output, target, data_range=1.0)

def ms_ssim_loss(output, target):
    return 1 - ms_ssim(output, target, data_range=1.0)

def crop_center(image, crop_size=256):
    h, w, c = image.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    # Crop the image (handle the color channel by slicing the last dimension)
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size, :]
    return cropped_image.astype(np.float32)

# Dataset for coordinate-pixel pairs
class SlideDataset(Dataset):
    def __init__(self, coordinates, pixels):
        self.coordinates = coordinates
        self.pixels = pixels

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return self.coordinates[idx], self.pixels[idx]

# Function to load and process SVS image
def load_svs_image(svs_path, level=0):
    slide = openslide.OpenSlide(svs_path)
    image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3] / 255.0
    slide.close()
    return image

# Prepare coordinate-pixel dataset
def create_dataset(image):
    h, w, _ = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coordinates = np.stack((x.ravel(), y.ravel()), axis=-1) / np.array([w, h])  # Normalize
    pixels = image.reshape(-1, 3)  # RGB values
    return coordinates.astype(np.float32), pixels.astype(np.float32)


def extract_patches(image, patch_size=64, stride=None):
    """
    Extract patches from an image with a given patch size and stride
    
    Args:
        image: Input image as numpy array (H, W, C)
        patch_size: Size of square patches to extract
        stride: Step size between patches (default: same as patch_size)
    
    Returns:
        patches: List of extracted patches
        positions: List of (x, y) positions of each patch
    """
    h, w, c = image.shape
    stride = stride or patch_size
    
    patches = []
    positions = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch)
            positions.append((x, y))
    
    return patches, positions

class PatchDataset(Dataset):
    """Dataset for image patches"""
    
    def __init__(self, patches, normalize=True):
        """
        Args:
            patches: List of image patches
            normalize: Whether to normalize pixel values to [-1, 1]
        """
        self.patches = patches
        self.normalize = normalize
        self.coordinates_list = []
        self.pixels_list = []
        
        # Pre-process all patches
        for patch in patches:
            coordinates, pixels = create_dataset(patch)
            self.coordinates_list.append(coordinates)
            if normalize:
                pixels = 2 * pixels - 1  # Normalize to [-1, 1]
            self.pixels_list.append(pixels)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.coordinates_list[idx], self.pixels_list[idx]


class PatchPositionDataset(Dataset):
    """Dataset that includes both patches and their positions"""
    
    def __init__(self, patches, positions, image_size, normalize=True):
        """
        Args:
            patches: List of image patches
            positions: List of (x, y) positions
            image_size: (width, height) of the original image
            normalize: Whether to normalize pixel values to [-1, 1]
        """
        self.patches = patches
        self.positions = positions
        self.image_size = image_size
        self.normalize = normalize
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        position = self.positions[idx]
        
        # Normalize positions to [0, 1]
        norm_position = torch.tensor([
            position[0] / self.image_size[0],
            position[1] / self.image_size[1]
        ], dtype=torch.float32)
        
        # Normalize patch to [-1, 1] if requested
        if self.normalize:
            patch = torch.tensor(patch, dtype=torch.float32)
            patch = 2.0 * patch - 1.0
        else:
            patch = torch.tensor(patch, dtype=torch.float32)
        
        # Convert to NCHW format
        patch = patch.permute(2, 0, 1)  # HWC to CHW
        
        return patch, norm_position

def train_autoencoder(patches, positions, image_size, batch_size=16, epochs=100):
    # Create dataset and dataloader
    dataset = PatchPositionDataset(patches, positions, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PositionalAutoencoder(in_channels=3, latent_dim=256, pos_dim=2).to(device)
    
    # Loss function and optimizer
    criterion = LPIPSWithL2Loss(l2_weight=0.95, perceptual_weight=0.05).to(device)
    #l2_criterion = torch.nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create output directory for samples
    os.makedirs("samples", exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_l2 = 0.0
        total_perceptual = 0.0
        psnr_values = []
        for batch_idx, (patches, positions) in enumerate(dataloader):
            patches = patches.to(device)
            positions = positions.to(device)
            
            # Forward pass
            reconstructions, latents = model(patches, positions)
            
            # Loss calculation
            loss, l2_loss, perceptual_loss = criterion(reconstructions, patches)
            #l2_loss = l2_criterion(reconstructions, patches)
            ms_ssim_loss_value = ssim(reconstructions, patches)
            #loss = 0.7 * loss + 0.3 * ms_ssim_loss_value

            # Record loss statistics
            total_loss += loss.item()
            total_l2 += l2_loss.item()
            total_perceptual += perceptual_loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            originals = (patches.detach().cpu().permute(0, 2, 3, 1).numpy() + 1) / 2
            recons = (reconstructions.detach().cpu().permute(0, 2, 3, 1).numpy() + 1) / 2
            psnr_values.append(psnr(originals, recons))

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, L2: {l2_loss.item():.4f}, "
                      f"Perceptual: {perceptual_loss.item():.4f}, "
                      f"MS-SSIM: {ms_ssim_loss_value.item():.4f}")
        
        # Print epoch statistics
        print(f"Epoch {epoch}, Avg Loss: {total_loss/len(dataloader):.4f}, "
              f"Avg L2: {total_l2/len(dataloader):.4f}, "
              f"Avg Perceptual: {total_perceptual/len(dataloader):.4f}")
        print(f"PSNR of reconstructed image: {sum(psnr_values) / len(psnr_values):.2f}")

        # Generate and save sample reconstructions
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                # Get a few samples
                sample_patches, sample_positions = next(iter(dataloader))
                sample_patches = sample_patches[:8].to(device)
                sample_positions = sample_positions[:8].to(device)
                
                # Get reconstructions
                reconstructions, _ = model(sample_patches, sample_positions)
                
                # Convert to numpy for saving (first denormalize from [-1, 1] to [0, 1])
                originals = (sample_patches.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2
                recons = (reconstructions.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2

                # Save samples
                for i in range(len(originals)):
                    imsave(f"samples/epoch_{epoch}_sample_{i}_original.png", 
                           (originals[i] * 255).astype(np.uint8))
                    imsave(f"samples/epoch_{epoch}_sample_{i}_recon.png", 
                           (recons[i] * 255).astype(np.uint8))
            

        # Save model checkpoint
        if epoch % 10 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"autoencoder_checkpoint_epoch_{epoch}.pth")

    return model, device

def reconstruct_image_from_patches(model, dataloader, image_size, device):
    width, height = image_size
        
    # Create empty canvas for the reconstructed image and weight map for blending
    reconstructed_image = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width, 1), dtype=np.float32)  # For blending overlapping regions

    for batch_idx, (patches, positions) in enumerate(dataloader):
        patches = patches.to(device)
        positions = positions.to(device)
        reconstructions, latents = model(patches, positions)
        
        # Move reconstructions back to CPU and convert to numpy for placement
        reconstructions = reconstructions.detach().cpu().numpy()
        positions = positions.detach().cpu().numpy()
        
        batch_size = reconstructions.shape[0]
        patch_height, patch_width = reconstructions.shape[2], reconstructions.shape[3]
        
        # Place each patch onto the canvas
        for i in range(batch_size):
            x, y = positions[i]  # Assuming positions are (x, y) coordinates of top-left corner
            print(f"Placing patch at position: ({x}, {y})")
            x, y = int(x * width), int(y * height)
            
            # Create weight mask for this patch (higher in center, lower at edges)
            # This helps with blending overlapping patches
            patch_weight = np.ones((patch_height, patch_width, 1), dtype=np.float32)
            
            # Optional: Create a smoother blend using a window function
            # For example, using a simple 2D distance from center:
            y_indices, x_indices = np.mgrid[0:patch_height, 0:patch_width]
            center_y, center_x = patch_height // 2, patch_width // 2
            dist = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
            max_dist = np.sqrt(center_y**2 + center_x**2)
            patch_weight = np.expand_dims(1.0 - (dist / max_dist), axis=2)
            
            # Get patch region on the canvas
            y_end = min(y + patch_height, height)
            x_end = min(x + patch_width, width)
            
            # Handle cases where patch might go outside the canvas
            y_patch_end = y_end - y
            x_patch_end = x_end - x
            
            # Skip if patch is completely outside
            if x >= width or y >= height or x_end <= 0 or y_end <= 0:
                continue
                
            # Handle partial patches at the edges
            if x < 0:
                x_patch_start = -x
                x = 0
            else:
                x_patch_start = 0
                
            if y < 0:
                y_patch_start = -y
                y = 0
            else:
                y_patch_start = 0
            
            # Add the patch to the reconstructed image with weights
            patch = reconstructions[i].transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
            patch_region = patch[y_patch_start:y_patch_end, x_patch_start:x_patch_end]
            weight_region = patch_weight[y_patch_start:y_patch_end, x_patch_start:x_patch_end]
            
            # Update the reconstructed image and weight map
            reconstructed_image[y:y_end, x:x_end] += patch_region * weight_region
            weight_map[y:y_end, x:x_end] += weight_region

    # Normalize by the weights to get the final image
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    reconstructed_image = reconstructed_image / (weight_map + epsilon)

    # Clip values to valid image range
    reconstructed_image = np.clip(reconstructed_image, 0, 1)

    return reconstructed_image


# Example usage in your main script:
if __name__ == "__main__":
    # This should come after you've extracted patches
    from final import extract_patches, load_svs_image
    
    # Load your image
    svs_path = "./data/test.svs"
    whole_image = load_svs_image(svs_path, level=3)
    
    # Extract patches
    patch_size = 64
    patches, positions = extract_patches(whole_image, patch_size=patch_size, stride=patch_size // 2)
    print(f"Extracted {len(patches)} patches of size {patch_size}x{patch_size}")
    
    # Get image dimensions
    h, w, _ = whole_image.shape
    
    # Train the autoencoder
    #model, device = train_autoencoder(patches, positions, image_size=(w, h), batch_size=64, epochs=40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PositionalAutoencoder(in_channels=3, latent_dim=256, pos_dim=2).to(device)
    model.load_state_dict(torch.load("autoencoder_checkpoint_epoch_30.pth")['model_state_dict'])

    patch_size = 64
    patches, positions = extract_patches(whole_image, patch_size=patch_size, stride=patch_size)
    reconstructed_image = reconstruct_image_from_patches(
        model, DataLoader(PatchPositionDataset(patches, positions, image_size=(w, h)), batch_size=4), 
        image_size=(w, h), device=device
    )

    # Save the reconstructed image
    output_path = "reconstructed_whole_image.png"
    imsave(output_path, (reconstructed_image * 255).astype(np.uint8))
    print(f"Reconstructed image saved to {output_path}")
    
    # Calculate metrics between original and reconstructed image
    whole_image_normalized = whole_image.astype(np.float32)
    img_psnr = psnr(whole_image_normalized, reconstructed_image)
    img_ssim = ssim(
        torch.from_numpy(whole_image_normalized).permute(2, 0, 1).unsqueeze(0), 
        torch.from_numpy(reconstructed_image).permute(2, 0, 1).unsqueeze(0),
        data_range=1.0
    )
    
    print(f"Whole Image PSNR: {img_psnr:.2f}")
    print(f"Whole Image SSIM: {img_ssim:.4f}")