import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imsave

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


# Main function
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process the image
    svs_path = "../data/test.svs"  # Replace with your file path
    whole_image = load_svs_image(svs_path,level=3)
    #image = load_svs_image(svs_path,level=3)

    # sample an impage 
    image = crop_center(whole_image, crop_size=512)


    # Save the original image
    imsave("original_image.png", (image * 255).astype(np.uint8))

    # Create dataset and dataloader
        # Extract patches from the whole image
    patch_size = 64
    patches, positions = extract_patches(whole_image, patch_size=patch_size, stride=patch_size)
    print(f"Extracted {len(patches)} patches of size {patch_size}x{patch_size}")
    
    # Create dataset from patches
    patch_dataset = PatchDataset(patches)
    
    # You can now use this dataset for training or create a DataLoader from it
    patch_dataloader = DataLoader(
        patch_dataset, 
        batch_size=1,  # Process one patch at a time
        shuffle=True,
        num_workers=4
    )

    # Initialize Positional Encoding
    pos_encoding = PositionalEncoding(num_encoding_functions=6, include_input=True).to(device)

    # Initialize INR model, loss, and optimizer
    #model = INR(2,64,3,positional_encoding=pos_encoding).to(device)
    model = CoordNet(2,3,num_res=1,positional_encoding=pos_encoding).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.999),weight_decay=1e-6)

    ckpt = torch.load("checkpoint_epoch_2500.pth")  # Replace with your file
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"Resumed training from epoch {start_epoch}")

    # Train the INR model
    epochs = 10000
    for epoch in range(start_epoch,epochs):
        model.train()
        epoch_loss = 0.0
        iteration = 0
        for coord_batch, pixel_batch in dataloader:
            coord_batch, pixel_batch = coord_batch.to(device), pixel_batch.to(device)
            optimizer.zero_grad()
            output = model(coord_batch)
            loss = criterion(output, pixel_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #if (iteration + 1) % 20 == 0:
            #    print(f"Epoch {epoch + 1}, Iteration {iteration + 1}, Loss: {loss.item():.6f}")
            iteration += 1 
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

        if (epoch + 1) % 100 == 0: 
            model.eval()
            with torch.no_grad():
                coords_tensor = torch.tensor(coordinates).to(device)
                reconstructed_pixels = model(coords_tensor).cpu().numpy()
                reconstructed_image = reconstructed_pixels.reshape(image.shape)
                psnr_value = psnr(image, reconstructed_image)
                print(f"PSNR of reconstructed image: {psnr_value:.2f}")

                reconstructed_image = (reconstructed_image+1)/2.0
                #reconstructed_image = ((reconstructed_image+1)/2.0)*(p_max-p_min)+p_min
                #print(reconstructed_image.shape, np.max(reconstructed_image), np.min(reconstructed_image))
                imsave("reconstructed_image_{:06}.png".format(epoch), (reconstructed_image * 255).astype(np.uint8))
            
            # free GPU memory 
            del coords_tensor, reconstructed_pixels 
            torch.cuda.empty_cache()

        # Save checkpoint
        if (epoch + 1) % 500 == 0:  # Save every 100 epochs, or adjust as needed
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader)
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")

        """
        # Load checkpoint
        checkpoint = torch.load("checkpoint_epoch_100.pth")  # Replace with your file
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch}")
        """ 
