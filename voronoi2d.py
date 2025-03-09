"""
2D Voronoi Model with Generic Property Vectors

This module implements a 2D representation based on Voronoi cells
with support for arbitrary property vectors per cell.
The implementation uses KD-trees for efficient region lookup and
numpy arrays instead of objects for better performance in optimization workflows.
"""

import numpy as np
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from typing import Tuple, Optional, List


class Voronoi2D:
    """
    2D Voronoi cell representation with generic property vectors.

    Each region has an associated vector of properties of dimension vector_dim.
    This generalizes the tissue model concept, allowing for any number of properties
    to be associated with each cell.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (0, 1, 0, 1),
        vector_dim: int = 2,
    ):
        """
        Initialize the Voronoi model.

        Args:
            bounds: The bounds of the space (xmin, xmax, ymin, ymax)
            vector_dim: Dimension of the property vector for each cell
        """
        self.bounds: Tuple[float, float, float, float] = bounds
        self.vector_dim: int = vector_dim
        self.centroids: Optional[torch.FloatTensor] = None
        self.property_vectors: Optional[torch.FloatTensor] = (
            None  # Shape: (n_cells, vector_dim)
        )
        self._kd_tree: Optional[KDTree] = None

    def find_regions(self, points: torch.FloatTensor) -> torch.LongTensor:
        """
        Find regions for query points. Note: this method converts
        torch tensors to numpy arrays for compatibility with scipy KD tree.

        Args:
            points: Array of shape (n, 2) containing n query points

        Returns:
            Array of region IDs corresponding to each point.
            -1 indicates out of bounds.
        """
        # Check points within bounds
        xmin, xmax, ymin, ymax = self.bounds
        in_bounds = torch.logical_and(
            torch.logical_and(points[:, 0] >= xmin, points[:, 0] <= xmax),
            torch.logical_and(points[:, 1] >= ymin, points[:, 1] <= ymax),
        )

        # Initialize results array with -1 (out of bounds)
        result = torch.full((len(points),), -1)

        # Process in-bounds points
        if torch.any(in_bounds):
            # get the in-bounds points and convert to numpy
            in_bounds_points = points[in_bounds].cpu().numpy()

            # Query the KD-tree for nearest neighbors
            _, indices = self._kd_tree.query(in_bounds_points, k=1)

            # Assign region IDs to in-bounds points
            result[in_bounds] = torch.tensor(
                indices, dtype=torch.long, device=points.device
            )

        return result

    def get_properties(self, points: torch.FloatTensor) -> torch.FloatTensor:
        """
        Get property vectors for multiple points at once.

        Args:
            points: Array of shape (n, 2) containing n query points

        Returns:
            Array of shape (n, vector_dim) containing the property vectors
            NaN values indicate out of bounds points.
        """
        region_ids = self.find_regions(points)
        valid_points = region_ids >= 0

        # Initialize result array with nans
        result = torch.full((len(points), self.vector_dim), torch.nan)

        # Set properties for valid points
        if torch.any(valid_points):
            result[valid_points] = self.property_vectors[region_ids[valid_points]]

        return result

    @classmethod
    def generate_random_model(
        cls,
        num_cells: int,
        bounds: Tuple[float, float, float, float] = (0, 1, 0, 1),
        vector_dim: int = 2,
        property_ranges: Optional[List[Tuple[float, float]]] = None,
        random_seed: Optional[int] = None,
    ) -> "Voronoi2D":
        """
        Generate a random Voronoi model with the specified number of cells.

        Args:
            num_cells: Number of Voronoi cells to generate
            bounds: The bounds of the space (xmin, xmax, ymin, ymax)
            vector_dim: Dimension of the property vector for each cell
            property_ranges: List of (min, max) ranges for each property dimension
                            (defaults to (0, 1) for all dimensions)
            random_seed: Random seed for reproducibility

        Returns:
            A new Voronoi model with random properties
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Default property ranges if not provided
        if property_ranges is None:
            property_ranges = [(0, 1)] * vector_dim

        # Ensure correct number of ranges
        if len(property_ranges) != vector_dim:
            raise ValueError(
                f"Expected {vector_dim} property ranges, got {len(property_ranges)}"
            )

        # Create new instance
        model = cls(bounds=bounds, vector_dim=vector_dim)

        xmin, xmax, ymin, ymax = bounds
        # Generate random centroids within the bounds using torch
        model.centroids = torch.rand(num_cells, 2)
        model.centroids[:, 0] = model.centroids[:, 0] * (xmax - xmin) + xmin
        model.centroids[:, 1] = model.centroids[:, 1] * (ymax - ymin) + ymin

        # Create property vectors
        model.property_vectors = torch.zeros((num_cells, vector_dim))

        # Fill each property dimension
        for i, (min_val, max_val) in enumerate(property_ranges):
            model.property_vectors[:, i] = (
                torch.rand(num_cells) * (max_val - min_val) + min_val
            )

        # Create KD-tree for efficient nearest neighbor search
        model._kd_tree = KDTree(model.centroids.cpu().numpy())

        return model

    def plot_voronoi_structure(
        self,
        ax: Optional[plt.Axes] = None,
        show_centroids: bool = True,
        color_cells: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the Voronoi structure.

        Args:
            ax: Matplotlib axis to plot on (creates new figure if None)
            show_centroids: Whether to show the cell centroids
            color_cells: colors each cell according to the color values
            given in the property vectors (assuming the last three values correspond to color)

        Returns:
            Matplotlib figure and axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        if self.centroids is None:
            raise ValueError("Voronoi model not yet generated")

        # Create a Voronoi diagram
        vor = Voronoi(self.centroids.detach().cpu().numpy())

        # Plot the basic Voronoi diagram
        voronoi_plot_2d(vor, ax=ax, show_points=show_centroids, show_vertices=False)

        # Set limits
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Voronoi Structure")

        # color cells
        if color_cells:
            print("Color cells!")
            for r in range(len(vor.point_region)):
                region = vor.regions[vor.point_region[r]]
                if not -1 in region:
                    polygon = [vor.vertices[i] for i in region]
                    plt.fill(
                        *zip(*polygon), color=self.property_vectors[r, -3:].numpy()
                    )

        return fig, ax

    def plot_property_map(
        self,
        property_idx: int = 0,
        ax: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        title: Optional[str] = None,
        show_colorbar: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a property map for the specified property dimension.

        Args:
            property_idx: Index of the property to plot (0 to vector_dim-1)
            ax: Matplotlib axis to plot on
            cmap: Colormap to use
            title: Title for the plot (defaults to "Property {property_idx}")

        Returns:
            Matplotlib figure and axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        if self.centroids is None:
            raise ValueError("Voronoi model not yet generated")

        if property_idx < 0 or property_idx >= self.vector_dim:
            raise ValueError(
                f"property_idx must be between 0 and {self.vector_dim - 1}"
            )

        # Get property values for the specified dimension
        values = self.property_vectors[:, property_idx].detach().cpu().numpy()

        if title is None:
            title = f"Property {property_idx}"

        # Create Voronoi diagram
        vor = Voronoi(self.centroids.detach().cpu().numpy())

        # Get Voronoi regions for coloring
        polygons = []
        region_colors = []

        # Loop through regions and create polygons
        for i, (region_idx, _) in enumerate(
            zip(vor.point_region, self.centroids.detach().cpu().numpy())
        ):
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:  # Skip unbounded or empty regions
                vertices = vor.vertices[region]
                polygon = plt.Polygon(vertices, closed=True)
                polygons.append(polygon)
                region_colors.append(values[i])

        # Create patch collection and add to plot
        pc = PatchCollection(
            polygons,
            cmap=plt.cm.get_cmap(cmap),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        pc.set_array(np.array(region_colors))
        ax.add_collection(pc)

        # Add a colorbar if requested
        if show_colorbar:
            plt.colorbar(pc, ax=ax, label=title)

        # Set limits and labels
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{title} Map")

        # Set fixed aspect ratio
        ax.set_aspect("equal", adjustable="box")

        return fig, ax

    def save_model(self, filepath: str) -> None:
        """
        Save the Voronoi model to a file.

        Args:
            filepath: Path to save the model to (.npz format)
        """
        torch.save(
            {
                "centroids": self.centroids,
                "property_vectors": self.property_vectors,
                "bounds": self.bounds,
                "vector_dim": self.vector_dim,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """
        Load a Voronoi model from a file.

        Args:
            filepath: Path to load the model from (.pt format)
        """
        data = torch.load(filepath, weights_only=False)
        self.centroids = data["centroids"]
        self.property_vectors = data["property_vectors"]
        self.bounds = data["bounds"]
        self.vector_dim = data["vector_dim"]

        # Regenerate KD-tree
        self._kd_tree = KDTree(self.centroids.cpu().numpy())

    def trace_ray(
        self,
        r0: torch.FloatTensor,
        rhat: torch.FloatTensor,
        num_points: int = 200,
        max_ray_length: int = 1000,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Trace a ray through the Voronoi tessellation and find all intersection points.

        This method implements ray tracing through a Voronoi tessellation to find
        the intersections between the ray and the boundaries of Voronoi cells.

        Parameters:
        r0 (torch.FloatTensor): The starting point of the ray (2D coordinates).
        rhat (torch.FloatTensor): The direction vector of the ray (2D coordinates).
        num_points (int): The number of points to sample along the ray.
        max_ray_length (int): maximum length of the ray

        Returns:
        Tuple containing:
            - torch.FloatTensor: Intersection distances (ell values)
            - torch.FloatTensor: Intersection points coordinates
            - torch.FloatTensor: Normal vectors at intersection points
        """
        if self.centroids is None:
            raise ValueError("Voronoi model not yet generated")

        # Normalize the ray direction
        rhat = rhat / torch.norm(rhat, p=2)

        # Generate points along the ray
        ells = torch.linspace(-1, max_ray_length, num_points).unsqueeze(1)
        rs = r0 + rhat * ells
        print(ells[-1])

        # Find regions for all sampling points
        region_ids = self.find_regions(rs)

        # Identify region changes to detect intersections
        # Only consider transitions between valid regions (exclude out-of-bounds)
        boundaries = torch.where(
            (region_ids[:-1] != -1)
            & (region_ids[1:] != -1)
            & (torch.diff(region_ids, dim=0) != 0)
        )[0]

        if len(boundaries) == 0:
            # No intersections found
            return (
                torch.tensor([]),
                torch.tensor([]).reshape(0, 2),
                torch.tensor([]).reshape(0, 2),
            )

        # Get the region IDs on both sides of each boundary
        region_ids1 = region_ids[boundaries]
        region_ids2 = region_ids[boundaries + 1]

        # Get centroids of intersecting regions
        mu1 = self.centroids[region_ids1]
        mu2 = self.centroids[region_ids2]

        # Calculate exact intersection points
        # The boundary between two Voronoi cells is the perpendicular bisector
        # of the line connecting their centroids.
        # For a point on this boundary: |p - μ₁|² = |p - μ₂|²
        # Substituting p = r₀ + ℓr̂ and solving for ℓ:
        numer = (
            torch.norm(mu1, dim=1, p=2) ** 2
            - torch.norm(mu2, dim=1, p=2) ** 2
            - 2 * torch.sum((mu1 - mu2) * r0, dim=1)
        )
        denom = 2 * torch.sum((mu1 - mu2) * rhat, dim=1)

        # Handle potential division by zero (ray parallel to boundary)
        valid_intersections = torch.abs(denom) > 1e-10

        if not torch.any(valid_intersections):
            return (
                torch.tensor([]),
                torch.tensor([]).reshape(0, 2),
                torch.tensor([]).reshape(0, 2),
            )

        # Filter out invalid intersections
        mu1 = mu1[valid_intersections]
        mu2 = mu2[valid_intersections]
        numer = numer[valid_intersections]
        denom = denom[valid_intersections]

        # Calculate ℓ values
        ells_intersect = numer / denom

        # Calculate intersection points
        intersection_points = r0 + rhat * ells_intersect.unsqueeze(1)

        # Calculate normal vectors at intersection points
        # The normal vector at the boundary is in the direction from μ₂ to μ₁
        normals = mu1 - mu2
        normals = normals / torch.norm(normals, dim=1, p=2).unsqueeze(1)

        # Sort results by ℓ value for consistent ordering
        sort_indices = torch.argsort(ells_intersect)
        ells_intersect = ells_intersect[sort_indices]
        intersection_points = intersection_points[sort_indices]
        normals = normals[sort_indices]

        return ells_intersect, intersection_points, normals


def demo() -> None:
    """Demo function showing the key features of the Voronoi model."""
    import time
    import os

    # Create output directory if it doesn't exist
    output_dir = "voronoi2d_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a 3-dimensional property vector model
    # properties: impedance, attenuation, and a new "elasticity" property
    model = Voronoi2D.generate_random_model(
        num_cells=1000,
        bounds=(0, 10, 0, 10),
        vector_dim=3,
        property_ranges=[(1.3, 1.7), (0.5, 1.5), (0.2, 0.8)],
        random_seed=42,
    )

    # Create a figure with 4 subplots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    # Plot Voronoi structure
    model.plot_voronoi_structure(ax=axs[0])
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_title("Voronoi Structure")

    # Plot property maps for each dimension with different colormaps
    model.plot_property_map(
        property_idx=0, ax=axs[1], cmap="viridis", title="Impedance"
    )
    axs[1].set_aspect("equal", adjustable="box")

    model.plot_property_map(
        property_idx=1, ax=axs[2], cmap="plasma", title="Attenuation"
    )
    axs[2].set_aspect("equal", adjustable="box")

    model.plot_property_map(
        property_idx=2, ax=axs[3], cmap="cividis", title="Elasticity"
    )
    axs[3].set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "voronoi2d_properties.png"), dpi=300)
    plt.close(fig)

    # Test query performance
    num_test_points = 100000
    test_points = torch.rand(num_test_points, 2) * 10

    start_time = time.time()
    properties = model.get_properties(test_points)
    batch_time = time.time() - start_time

    print(f"Batch query time ({num_test_points} points): {batch_time:.4f} seconds")
    print(f"Property vector shape for queries: {properties.shape}")

    # Test serialization
    model_path = os.path.join(output_dir, "voronoi2d_model.npz")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Test loading
    new_model = Voronoi2D(vector_dim=3)
    new_model.load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Delete the file
    os.remove(model_path)
    print(f"Model file {model_path} deleted")


if __name__ == "__main__":
    demo()
