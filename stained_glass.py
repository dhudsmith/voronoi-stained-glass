#!/usr/bin/env python3
"""
Stained Glass Voronoi Demo Script

This script optimizes a Voronoi2D instance (centroids and RGB properties)
to match a reference image, creating a stained glass effect. It uses a
k-nearest neighbor weighted approach to generate smooth color blending
between Voronoi cells.

Usage:
    python stained_glass.py <image_path> [--config CONFIG_PATH]
"""

import math
import argparse
import time
from pathlib import Path
import yaml
from typing import Tuple, List, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial import KDTree
import shutil

from pytorch_msssim import MS_SSIM

from voronoi2d import Voronoi2D


class StainedGlassVoronoi:
    """
    Class for the stained glass Voronoi optimization demo.

    This class handles loading an image, initializing a Voronoi model,
    optimizing the model to match the image, and generating visualizations.
    Preserves the aspect ratio of the original image.
    """

    def __init__(self, config: Dict[str, Any], image_path: Optional[str] = None):
        """
        Initialize the demo with configuration.

        Args:
            config: Dictionary containing configuration parameters
            image_path: Path to the target image
        """
        self.config = config
        self.image_path = image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize empty properties
        self.target_image: Optional[torch.Tensor] = None
        self.model: Optional[Voronoi2D] = None
        self.output_dir: Optional[Path] = None
        self.rendered_dir: Optional[Path] = None
        self.centroids_dir: Optional[Path] = None
        self.comparison_dir: Optional[Path] = None
        self.models_dir: Optional[Path] = None

        # Image dimensions (will be set in load_image)
        self.image_width: int = 0
        self.image_height: int = 0

        # Create directory structure
        self.setup_directories()

    def setup_directories(self) -> None:
        """
        Create output directory structure for saving results.
        """
        self.output_dir = Path(self.config["output"]["dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create subfolders for organizing different outputs
        self.rendered_dir = self.output_dir / "rendered"
        self.centroids_dir = self.output_dir / "centroids"
        self.comparison_dir = self.output_dir / "comparison"
        self.models_dir = self.output_dir / "models"

        for directory in [
            self.rendered_dir,
            self.centroids_dir,
            self.comparison_dir,
            self.models_dir,
        ]:
            directory.mkdir(exist_ok=True, parents=True)

        print(f"Output directories created at {self.output_dir}")

    def load_image(self) -> None:
        """
        Load and preprocess reference image while preserving aspect ratio.
        """
        if not self.image_path:
            raise ValueError("No image path provided")

        # Load image
        image = Image.open(self.image_path).convert("RGB")
        original_width, original_height = image.size

        # Determine if we should force square or preserve aspect ratio
        force_square = self.config["image"].get("force_square", False)
        min_dimension = self.config["image"]["min_dimension"]

        if force_square:
            # Force square image as before - using min_dimension as the size
            self.image_width = min_dimension
            self.image_height = min_dimension
            transform = transforms.Compose(
                [
                    transforms.Resize((min_dimension, min_dimension)),
                    transforms.ToTensor(),
                ]
            )
        else:
            # Preserve aspect ratio
            # Calculate new dimensions based on minimum dimension while preserving aspect ratio
            scaling_factor = min_dimension / min(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)

            self.image_width = new_width
            self.image_height = new_height

            transform = transforms.Compose(
                [transforms.Resize((new_height, new_width)), transforms.ToTensor()]
            )

        # Load and move to device
        self.target_image = transform(image).to(self.device)

        print(
            f"Loaded image from {self.image_path} and resized to {self.image_width}x{self.image_height}"
        )
        print(
            f"Aspect ratio: {self.image_width / self.image_height:.2f} (original: {original_width / original_height:.2f})"
        )

    def initialize_model(self) -> None:
        """
        Initialize the Voronoi model with random centroids and color properties.
        Uses actual image dimensions for bounds rather than assuming square.
        """
        num_cells = self.config["voronoi"]["num_cells"]
        random_seed = self.config["random_seed"]

        # Use actual image dimensions for bounds
        # Note: bounds are 0-indexed
        bounds = (0, self.image_width - 1, 0, self.image_height - 1)

        # Generate random Voronoi model
        self.model = Voronoi2D.generate_random_model(
            num_cells=num_cells,
            bounds=bounds,
            vector_dim=3,  # RGB colors
            property_ranges=[
                (0, 1),
                (0, 1),
                (0, 1),
            ],  # [0,1] range for each RGB channel
            random_seed=random_seed,
        )

        # Move model parameters to device
        self.model.centroids = self.model.centroids.to(self.device)
        self.model.property_vectors = self.model.property_vectors.to(self.device)

        print(f"Initialized Voronoi model with {num_cells} cells and bounds {bounds}")

    # ----- Rendering Methods -----

    def compute_weighted_colors(
        self,
        points: torch.Tensor,
        centroids: torch.Tensor,
        properties: torch.Tensor,
        k: int,
        sigma: float,
    ) -> torch.Tensor:
        """
        Compute weighted colors based on k nearest centroids.

        Args:
            points: Tensor of shape (n, 2) with point coordinates
            centroids: Tensor of shape (m, 2) with centroid coordinates
            properties: Tensor of shape (m, 3) with RGB colors
            k: Number of nearest neighbors to consider
            sigma: Temperature parameter for softmax

        Returns:
            Tensor of shape (n, 3) with weighted colors
        """
        # Compute squared distances between points and centroids
        n_points = points.shape[0]
        n_centroids = centroids.shape[0]

        # Reshape for broadcasting
        points_expanded = points.unsqueeze(1)  # (n, 1, 2)
        centroids_expanded = centroids.unsqueeze(0)  # (1, m, 2)

        # Compute squared distances
        distances_squared = torch.sum(
            (points_expanded - centroids_expanded) ** 2, dim=2
        )  # (n, m)

        # Get top-k nearest centroids (smallest distances)
        distances_k, indices_k = torch.topk(
            distances_squared, k=min(k, n_centroids), dim=1, largest=False
        )

        # Apply softmax weighting
        weights = F.softmax(-distances_k / (sigma**2), dim=1)  # (n, k)

        # Get properties of the k nearest centroids
        selected_props = properties[indices_k]  # (n, k, 3)

        # Weighted sum of properties
        weighted_colors = torch.sum(
            selected_props * weights.unsqueeze(-1), dim=1
        )  # (n, 3)

        return weighted_colors

    def render_voronoi_image(
        self, width: int, height: int, k_neighbors: int, sigma: float
    ) -> torch.Tensor:
        """
        Render an image from the Voronoi model using weighted colors.
        Now supports rectangular (non-square) dimensions.

        Args:
            width: Width of the output image
            height: Height of the output image
            k_neighbors: Number of nearest neighbors for color weighting
            sigma: Temperature parameter

        Returns:
            Tensor of shape (3, height, width) representing the rendered image
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        # Create a grid of points covering the image
        x = torch.linspace(0, width - 1, width, device=self.device)
        y = torch.linspace(0, height - 1, height, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=1
        )  # (width*height, 2)

        # Compute weighted colors for all points
        weighted_colors = self.compute_weighted_colors(
            points,
            self.model.centroids,
            self.model.property_vectors,
            k_neighbors,
            sigma,
        )

        # Reshape to image format
        return weighted_colors.reshape(height, width, 3).permute(
            2, 0, 1
        )  # (3, height, width)

    def render_voronoi_nearest_neighbor(
        self, width: int, height: int, upsample_ratio: int = 10
    ) -> torch.Tensor:
        """
        Render an image from the Voronoi model using only the nearest neighbor.
        This produces a true Voronoi diagram with hard boundaries.
        Now supports rectangular (non-square) dimensions.

        Args:
            width: Width of the output image
            height: Height of the output image
            upsample_ratio: Ratio to upsample the grid for finer rendering

        Returns:
            Tensor of shape (3, height, width) representing the rendered image
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        # Create a grid of points covering the image with upsampled resolution
        upsampled_width = width * upsample_ratio
        upsampled_height = height * upsample_ratio

        x = torch.linspace(0, width - 1, upsampled_width, device=self.device)
        y = torch.linspace(0, height - 1, upsampled_height, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        points = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=1
        )  # (upsampled_width*upsampled_height, 2)

        # Find the nearest centroid for each point
        region_ids = self.model.find_regions(points)

        # Get the color for each region
        colors = self.model.property_vectors[region_ids]

        # Reshape to upsampled image format
        upsampled_image = colors.reshape(upsampled_height, upsampled_width, 3).permute(
            2, 0, 1
        )  # (3, upsampled_height, upsampled_width)

        return upsampled_image

    # ----- Optimization Methods -----

    def optimize(self) -> List[Tuple[float, float, float]]:
        """
        Run the optimization process to match the Voronoi model to the target image.

        Returns:
            List of (MSE, SSIM, Combined Loss) tuples for each iteration
        """
        if self.model is None or self.target_image is None:
            raise ValueError(
                "Model and target image must be initialized before optimization"
            )

        # Copy parameters and enable gradient tracking
        centroids = self.model.centroids.clone().requires_grad_(True)
        properties = self.model.property_vectors.clone().requires_grad_(True)

        # Set up optimizer
        lr = self.config["optimization"]["learning_rate"]
        optimizer = optim.Adam([centroids, properties], lr=lr)

        # Set up loss functions
        mse_loss = nn.MSELoss()

        # Initialize the SSIM module from pytorch-msssim
        ssim_module = MS_SSIM(
            win_size=self.config["optimization"]["ssim_window_size"], size_average=True
        ).to(self.device)

        # Set up tracking variables
        iterations = self.config["optimization"]["iterations"]
        ssim_weight = self.config["optimization"]["ssim_weight"]
        save_every = self.config["optimization"]["save_every"]
        k_neighbors = self.config["voronoi"]["k_neighbors"]

        # Initial sigma is now based on the average of width and height
        initial_sigma = max(self.image_width, self.image_height) * 0.1

        start_time = time.time()
        loss_history = []

        # Create a log file to track progress
        with open(self.output_dir / "results.txt", "w") as f:
            f.write(f"Optimizing Voronoi model to match {self.image_path}\n")
            f.write("Parameters: " + yaml.dump(self.config, default_flow_style=False))
            f.write("\nIteration,MSE,SSIM,Combined_Loss,Time(s)\n")

        # Calculate weight decay rate (so sigma is 10% of initial value by end)
        rate = -math.log(self.config["optimization"]["sigma_final_ratio"])

        # Main optimization loop
        for iteration in tqdm(range(iterations)):
            # Update sigma (temperature) - exponential annealing
            current_sigma = initial_sigma * math.exp(-iteration / iterations * rate)

            # Zero gradients
            optimizer.zero_grad()

            # Update model with current parameters
            self.model.centroids = centroids
            self.model.property_vectors = properties

            # Important: Update the KD-tree with new centroid positions
            self.model._kd_tree = KDTree(self.model.centroids.detach().cpu().numpy())

            # Render image from model - using actual width and height
            rendered_image = self.render_voronoi_image(
                self.image_width, self.image_height, k_neighbors, current_sigma
            )

            # Compute losses
            mse = mse_loss(rendered_image, self.target_image)

            # Calculate SSIM value (1 = perfect similarity)
            # Add batch dimension for SSIM computation
            ssim_value = ssim_module(
                rendered_image.unsqueeze(0), self.target_image.unsqueeze(0)
            )

            # Convert to loss (1 - SSIM) since we want to minimize
            ssim_loss = 1.0 - ssim_value

            # Combine losses
            loss = (1 - ssim_weight) * mse + ssim_weight * ssim_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Ensure properties stay in valid range [0,1]
            with torch.no_grad():
                properties.clamp_(0, 1)
                # Ensure centroids stay within bounds
                centroids[:, 0].clamp_(0, self.image_width - 1)
                centroids[:, 1].clamp_(0, self.image_height - 1)

            # Store loss values
            loss_val = loss.item()
            mse_val = mse.item()
            ssim_val = ssim_loss.item()  # We store the loss value (1 - SSIM)
            loss_history.append((mse_val, ssim_val, loss_val))

            # Save visualizations periodically
            if iteration % save_every == 0 or iteration == iterations - 1:
                elapsed = time.time() - start_time

                # Log to file
                with open(self.output_dir / "results.txt", "a") as f:
                    f.write(
                        f"{iteration},{mse_val:.6f},{ssim_val:.6f},{loss_val:.6f},{elapsed:.2f}\n"
                    )

                # Save visualizations
                self.save_visualizations(
                    iteration=iteration,
                    rendered_image=rendered_image,
                    centroids=centroids,
                    properties=properties,
                    loss_values=(mse_val, ssim_val, loss_val),
                    elapsed=elapsed,
                    sigma=current_sigma,
                )

        # Final update of model parameters
        self.model.centroids = centroids.detach().clone()
        self.model.property_vectors = properties.detach().clone()

        # Save final model
        self.model.save_model(self.models_dir / "voronoi_model_final.pt")

        # Generate final visualizations
        self.generate_final_visualizations()

        return loss_history

    # ----- Visualization Methods -----

    def save_visualizations(
        self,
        iteration: int,
        rendered_image: torch.Tensor,
        centroids: torch.Tensor,
        properties: torch.Tensor,
        loss_values: Tuple[float, float, float],
        elapsed: float,
        sigma: float,
    ) -> None:
        """
        Save various visualizations at a given iteration.

        Args:
            iteration: Current iteration number
            rendered_image: The rendered image tensor
            centroids: Current centroid positions
            properties: Current property vectors
            loss_values: Tuple of (MSE, SSIM, combined loss)
            elapsed: Elapsed time in seconds
            sigma: Current sigma value
        """
        mse_val, ssim_val, loss_val = loss_values

        # Save current rendered image
        save_image(rendered_image, self.rendered_dir / f"rendered_{iteration:04d}.png")

        # Visualize Voronoi structure (boundaries only)
        plt.figure(
            figsize=(10, 10 * self.image_height / self.image_width)
        )  # Preserve aspect ratio in figures

        # Use Voronoi2D's built-in structure plotting
        fig, ax = self.model.plot_voronoi_structure(show_centroids=False)

        # Overlay centroids with their colors
        plt.scatter(
            centroids.detach().cpu().numpy()[:, 0],
            centroids.detach().cpu().numpy()[:, 1],
            c=properties.detach().cpu().numpy(),
            s=50,
        )

        plt.xlim(0, self.image_width - 1)
        plt.ylim(self.image_height - 1, 0)  # Flip the display vertically
        plt.gca().set_aspect("equal")
        plt.gca().axis("off")  # Hide axes
        plt.title(f"Voronoi Structure - Iteration {iteration}")
        plt.savefig(self.centroids_dir / f"centroids_{iteration:04d}.png")
        plt.close()

        # Plot side by side comparison
        # Adjust figure size to maintain the image aspect ratio
        plt.figure(figsize=(12, 6 * self.image_height / self.image_width))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.target_image.permute(1, 2, 0).cpu().numpy())
        plt.title("Target Image")
        plt.axis("off")

        # Rendered image
        plt.subplot(1, 2, 2)
        plt.imshow(rendered_image.permute(1, 2, 0).detach().cpu().numpy())
        plt.title(f"Voronoi Approximation (Iteration {iteration})")
        plt.axis("off")

        plt.savefig(self.comparison_dir / f"comparison_{iteration:04d}.png")
        plt.close()

        # Print status
        print(f"Iteration {iteration}/{self.config['optimization']['iterations']}")
        print(f"  MSE: {mse_val:.6f}, SSIM Loss: {ssim_val:.6f}, Loss: {loss_val:.6f}")
        print(f"  Sigma: {sigma:.2f}")
        print(f"  Time elapsed: {elapsed:.2f}s")

    def generate_final_visualizations(self) -> None:
        """
        Generate final visualizations after optimization is complete.
        """
        # Save final rendered image using weighted KNN
        rendered_image = self.render_voronoi_image(
            self.image_width,
            self.image_height,
            self.config["voronoi"]["k_neighbors"],
            max(self.image_width, self.image_height)
            * 0.01,  # Small sigma for cleaner boundaries
        )
        save_image(rendered_image, self.rendered_dir / "rendered_final.png")

        # Save final image using nearest neighbor only (true Voronoi diagram)
        nearest_neighbor_image = self.render_voronoi_nearest_neighbor(
            self.image_width, self.image_height
        )
        save_image(
            nearest_neighbor_image,
            self.output_dir / f"voronoi_nearest_neighbor_final.png",
        )

        # Create a side-by-side comparison of original and NN Voronoi
        # Adjust figure size to maintain aspect ratio
        plt.figure(figsize=(12, 6 * self.image_height / self.image_width))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.target_image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        # Nearest neighbor Voronoi
        plt.subplot(1, 2, 2)
        plt.imshow(nearest_neighbor_image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(self.output_dir / "final_comparison.png")
        plt.close()

    def plot_loss_history(self, loss_history: List[Tuple[float, float, float]]) -> None:
        """
        Plot loss history over iterations.

        Args:
            loss_history: List of (MSE, SSIM, combined loss) tuples
        """
        plt.figure(figsize=(10, 6))
        iterations = list(range(len(loss_history)))
        plt.plot(iterations, [l[0] for l in loss_history], label="MSE")
        plt.plot(iterations, [l[1] for l in loss_history], label="SSIM Loss")
        plt.plot(iterations, [l[2] for l in loss_history], label="Combined")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.output_dir / "loss_history.png")
        plt.close()

    def create_gifs(self) -> None:
        """
        Create GIFs from the saved images.
        """
        print("Creating GIFs from saved images...")

        # Helper function to create GIF from images in a directory
        def create_gif(image_dir: Path, output_path: Path, duration: int = 100) -> None:
            """Create a GIF from a sequence of images."""
            images = sorted(list(image_dir.glob("*.png")))
            if not images:
                print(f"No images found in {image_dir}")
                return

            frames = [Image.open(image) for image in images]
            # Save the first frame as a GIF with the rest of the frames appended
            frames[0].save(
                output_path,
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=duration,  # milliseconds per frame
                loop=0,  # 0 means loop forever
            )
            print(f"Created GIF: {output_path}")

        # Create GIFs for each directory
        create_gif(self.centroids_dir, self.output_dir / "centroids_evolution.gif")
        create_gif(self.comparison_dir, self.output_dir / "comparison_evolution.gif")

        # Optionally remove the individual frames to save space
        if self.config.get("output", {}).get("clean_temp_files", True):
            print("Cleaning up temporary image files...")
            for directory in [
                self.rendered_dir,
                self.centroids_dir,
                self.comparison_dir,
            ]:
                shutil.rmtree(directory)
                print(f"Removed {directory}")

    # ----- Main Execution Method -----

    def run(self) -> None:
        """
        Run the full stained glass demo process.
        """
        print(f"Starting Stained Glass Voronoi optimization")
        print(f"Using device: {self.device}")

        start_time = time.time()

        # Load reference image
        self.load_image()

        # Initialize Voronoi model
        self.initialize_model()

        # Run optimization
        loss_history = self.optimize()

        # Plot loss history
        self.plot_loss_history(loss_history)

        # Create GIFs
        self.create_gifs()

        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f}s")
        print(f"Results saved to {self.output_dir}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Optimize Voronoi tessellation to match a reference image."
    )
    parser.add_argument("image_path", type=str, help="Path to reference image")
    parser.add_argument(
        "--config",
        type=str,
        default="stained_glass_config.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file or create default if it doesn't exist.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

            # Handle backward compatibility for 'size' and 'max_dimension' parameters
            if "image" in config:
                if "size" in config["image"]:
                    size = config["image"].pop("size")
                    if "min_dimension" not in config["image"]:
                        config["image"]["min_dimension"] = size
                    print(
                        f"Converted legacy 'size' parameter to 'min_dimension': {size}"
                    )
                elif "max_dimension" in config["image"]:
                    max_dim = config["image"].pop("max_dimension")
                    if "min_dimension" not in config["image"]:
                        config["image"]["min_dimension"] = max_dim
                    print(
                        f"Converted legacy 'max_dimension' parameter to 'min_dimension': {max_dim}"
                    )

            # Ensure force_square has a default value
            if "image" in config and "force_square" not in config["image"]:
                config["image"]["force_square"] = False

            return config

    # Default configuration
    default_config = {
        "image": {
            "min_dimension": 128,  # Min dimension for resizing while preserving aspect ratio
            "force_square": False,  # Whether to force images to be square
        },
        "voronoi": {
            "num_cells": 100,  # Number of Voronoi cells
            "k_neighbors": 3,  # Number of neighbors for color weighting
        },
        "optimization": {
            "iterations": 200,  # Number of optimization iterations
            "learning_rate": 0.1,  # Learning rate for Adam optimizer
            "ssim_weight": 0.5,  # Weight for SSIM in loss calculation (0-1)
            "save_every": 20,  # Save intermediate results every N iterations
            "sigma_final_ratio": 0.1,  # Final sigma value as a ratio of initial
            "ssim_window_size": 5,  # Window size for SSIM calculation
        },
        "output": {
            "dir": "voronoi_image_output",  # Output directory
            "clean_temp_files": True,  # Whether to clean up temp files after GIF creation
        },
        "random_seed": 42,  # Random seed for reproducibility
    }

    # Save default config
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    print(f"Created default configuration file at {config_path}")
    return default_config


def main() -> None:
    """
    Main function that parses arguments and runs the demo.
    """
    args = parse_args()
    config = load_config(args.config)

    # Set random seed for reproducibility
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # Run demo
    demo = StainedGlassVoronoi(config, args.image_path)
    demo.run()


if __name__ == "__main__":
    main()
