import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import glob
import argparse
from tqdm import tqdm

# Import required metrics libraries
import lpips
from skimage.metrics import structural_similarity as ssim
import clip
from pytorch_fid import fid_score

class VTONMetricsCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Initialize CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
    
    def load_and_preprocess_image(self, image_path, size=(256, 192)):
        """Load and preprocess image for metric computation"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        
        # Convert to tensor for LPIPS and CLIP
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Also return numpy array for SSIM
        img_np = np.array(img) / 255.0
        
        return img_tensor, img_np, img
    
    def compute_lpips(self, img1_tensor, img2_tensor):
        """Compute LPIPS score"""
        with torch.no_grad():
            lpips_score = self.lpips_model(img1_tensor, img2_tensor)
        return lpips_score.item()
    
    def compute_ssim(self, img1_np, img2_np):
        """Compute SSIM score"""
        return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    
    def compute_clip_similarity(self, img1, img2):
        """Compute CLIP image similarity"""
        img1_clip = self.clip_preprocess(img1).unsqueeze(0).to(self.device)
        img2_clip = self.clip_preprocess(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image1_features = self.clip_model.encode_image(img1_clip)
            image2_features = self.clip_model.encode_image(img2_clip)
            
            # Normalize features
            image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
            image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = torch.sum(image1_features * image2_features).item()
        
        return similarity
    
    def compute_fid(self, real_images_path, generated_images_path, batch_size=32):
        """Compute FID score between real and generated images"""
        fid_value = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path],
                                                       batch_size=batch_size,
                                                       device=self.device,
                                                       dims=2048)
        return fid_value
    
    def evaluate_dataset(self, ground_truth_dir, generated_dir, output_file=None):
        """Evaluate all metrics on a dataset"""
        lpips_scores = []
        ssim_scores = []
        clip_scores = []
        
        # Get all image pairs
        gt_images = sorted(glob.glob(os.path.join(ground_truth_dir, '*.jpg')) + 
                          glob.glob(os.path.join(ground_truth_dir, '*.png')))
        gen_images = sorted(glob.glob(os.path.join(generated_dir, '*.jpg')) + 
                           glob.glob(os.path.join(generated_dir, '*.png')))
        
        # Ensure same number of images
        assert len(gt_images) == len(gen_images), f"Mismatch in number of images: {len(gt_images)} vs {len(gen_images)}"
        
        print(f"Evaluating {len(gt_images)} image pairs...")
        
        for gt_path, gen_path in tqdm(zip(gt_images, gen_images), total=len(gt_images)):
            # Load images
            gt_tensor, gt_np, gt_pil = self.load_and_preprocess_image(gt_path)
            gen_tensor, gen_np, gen_pil = self.load_and_preprocess_image(gen_path)
            
            # Compute metrics
            lpips_score = self.compute_lpips(gt_tensor, gen_tensor)
            ssim_score = self.compute_ssim(gt_np, gen_np)
            clip_score = self.compute_clip_similarity(gt_pil, gen_pil)
            
            lpips_scores.append(lpips_score)
            ssim_scores.append(ssim_score)
            clip_scores.append(clip_score)
        
        # Compute FID
        print("Computing FID score...")
        fid_value = self.compute_fid(ground_truth_dir, generated_dir)
        
        # Calculate averages
        results = {
            'LPIPS': np.mean(lpips_scores),
            'SSIM': np.mean(ssim_scores),
            'CLIP-Image Similarity': np.mean(clip_scores),
            'FID': fid_value
        }
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(f"LPIPS ↓: {results['LPIPS']:.4f}")
        print(f"SSIM ↑: {results['SSIM']:.4f}")
        print(f"CLIP-Image Similarity ↑: {results['CLIP-Image Similarity']:.4f}")
        print(f"FID ↓: {results['FID']:.2f}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Metric,Value\n")
                for metric, value in results.items():
                    f.write(f"{metric},{value}\n")
            print(f"\nResults saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Compute VTON evaluation metrics')
    parser.add_argument('--ground_truth', type=str, required=True, help='Directory containing ground truth images')
    parser.add_argument('--generated', type=str, required=True, help='Directory containing generated images')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    calculator = VTONMetricsCalculator(device=args.device)
    results = calculator.evaluate_dataset(args.ground_truth, args.generated, args.output)


if __name__ == '__main__':
    main()