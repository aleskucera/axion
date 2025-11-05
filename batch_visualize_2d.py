#!/usr/bin/env python3
"""
Batch 2D Loss Landscape Visualization Script

This script processes an entire simulation HDF5 file and creates 2D matplotlib plots
for all available timesteps, organizing them in a directory structure.

Usage:
    python batch_visualize_2d.py <hdf5_file> [options]

Examples:
    # Basic batch processing - creates plots in ./visualization_output/
    python batch_visualize_2d.py simulation_data.h5

    # Custom output directory
    python batch_visualize_2d.py simulation_data.h5 --output-dir my_plots/

    # Process specific range of timesteps
    python batch_visualize_2d.py simulation_data.h5 --start 10 --end 50

    # Process every N-th timestep (for large simulations)
    python batch_visualize_2d.py simulation_data.h5 --step-interval 5

    # High resolution plots
    python batch_visualize_2d.py simulation_data.h5 --dpi 300 --figsize 16 12
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from axion.logging.hdf5_reader import HDF5Reader

# Import the visualizer class
from visualize_loss_landscape import LossLandscapeVisualizer


class BatchVisualizer2D:
    """Batch processor for creating 2D loss landscape visualizations."""

    def __init__(self, hdf5_path: str, output_dir: str = "visualization_output"):
        """Initialize batch visualizer.

        Args:
            hdf5_path: Path to HDF5 file with simulation data
            output_dir: Directory to save all plots
        """
        self.hdf5_path = hdf5_path
        self.output_dir = Path(output_dir)
        self.visualizer = LossLandscapeVisualizer(hdf5_path)

        # Create output directory structure
        self.setup_output_directory()

    def setup_output_directory(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "thumbnails").mkdir(exist_ok=True)

        print(f"📁 Output directory: {self.output_dir.absolute()}")

    def create_enhanced_2d_plot(
        self,
        data: dict,
        step: int,
        output_path: str,
        figsize: tuple = (12, 9),
        dpi: int = 150,
        show_trajectory_numbers: bool = True,
    ) -> str:
        """Create enhanced 2D plot with additional customization options."""

        loss_grid = data["residual_norm_grid"]
        alphas = data["pca_alphas"]
        betas = data["pca_betas"]
        trajectory_2d = data["trajectory_2d_projected"]

        fig, ax = plt.subplots(figsize=figsize)

        # Clip small values to prevent log(0) errors
        loss_grid_clipped = np.clip(loss_grid, 1e-12, None)

        # Enhanced contour plot with more levels and better colormap
        levels = np.logspace(
            np.log10(loss_grid_clipped.min()), np.log10(loss_grid_clipped.max()), 25
        )

        contour = ax.contourf(
            alphas,
            betas,
            loss_grid_clipped.T,
            levels=levels,
            cmap="plasma",
            norm=plt.matplotlib.colors.LogNorm(),
        )

        # Add contour lines for better structure visibility
        contour_lines = ax.contour(
            alphas,
            betas,
            loss_grid_clipped.T,
            levels=levels[::4],  # Fewer contour lines for cleaner look
            colors="white",
            alpha=0.4,
            linewidths=0.8,
        )

        # Enhanced colorbar with better tick formatting
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Residual Norm", fontsize=14, weight="bold")

        # Format colorbar ticks in scientific notation
        import matplotlib.ticker as ticker

        cbar.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: f"{x:.1e}" if x != 0 else "0")
        )

        # Add more ticks to colorbar for better readability
        tick_locs = np.logspace(
            np.log10(loss_grid_clipped.min()), np.log10(loss_grid_clipped.max()), 8
        )
        cbar.set_ticks(tick_locs)

        if len(trajectory_2d) > 1:
            # Plot trajectory
            ax.plot(
                trajectory_2d[:, 0],
                trajectory_2d[:, 1],
                "w-",
                linewidth=2.5,
                alpha=0.9,
                label="Optimization Path",
                zorder=10,
            )

            # Mark iteration points
            if show_trajectory_numbers:
                step_size = max(1, len(trajectory_2d) // 8)  # Show max 8 numbers
                for i in range(0, len(trajectory_2d), step_size):
                    ax.plot(
                        trajectory_2d[i, 0],
                        trajectory_2d[i, 1],
                        "wo",
                        markersize=8,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        zorder=15,
                    )
                    ax.annotate(
                        str(i),
                        (trajectory_2d[i, 0], trajectory_2d[i, 1]),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=10,
                        color="white",
                        weight="bold",
                        zorder=20,
                    )

            # Highlight start and end
            ax.plot(
                trajectory_2d[0, 0],
                trajectory_2d[0, 1],
                "go",
                markersize=12,
                label="Start",
                markeredgecolor="black",
                markeredgewidth=2,
                zorder=15,
            )
            ax.plot(
                trajectory_2d[-1, 0],
                trajectory_2d[-1, 1],
                "rx",
                markersize=14,
                markeredgewidth=3,
                label="Solution",
                zorder=15,
            )

        # Enhanced styling
        ax.set_title(f"Loss Landscape - Timestep {step}", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Principal Component 1", fontsize=14, weight="bold")
        ax.set_ylabel("Principal Component 2", fontsize=14, weight="bold")
        ax.legend(framealpha=0.9, fontsize=12, loc="best")
        ax.grid(True, linestyle="--", alpha=0.3)

        # Add metadata if available
        if "pca_metadata" in data:
            metadata = data["pca_metadata"]
            info_text = (
                f"Grid: {int(metadata[0])}×{int(metadata[0])}\n"
                f"Iterations: {int(metadata[2])}\n"
                f"Timestep: {step}"
            )
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                fontsize=10,
                zorder=25,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

        return output_path

    def create_thumbnail(self, main_plot_path: str, thumbnail_path: str, size: tuple = (300, 225)):
        """Create a thumbnail version of the plot."""
        from PIL import Image

        try:
            with Image.open(main_plot_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, "PNG", optimize=True)
        except ImportError:
            print("⚠️  PIL not available - skipping thumbnail creation")
        except Exception as e:
            print(f"⚠️  Failed to create thumbnail for {main_plot_path}: {e}")

    def process_all_timesteps(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        step_interval: int = 1,
        figsize: tuple = (12, 9),
        dpi: int = 150,
        create_thumbnails: bool = True,
        show_trajectory_numbers: bool = True,
    ) -> List[str]:
        """Process all timesteps and create 2D plots.

        Args:
            start_step: First timestep to process (None = from beginning)
            end_step: Last timestep to process (None = to end)
            step_interval: Process every N-th timestep
            figsize: Figure size for plots
            dpi: DPI for saved plots
            create_thumbnails: Whether to create thumbnail versions
            show_trajectory_numbers: Whether to show iteration numbers on trajectory

        Returns:
            List of created file paths
        """
        # Get available timesteps
        all_steps = self.visualizer.get_available_steps()

        if not all_steps:
            print("❌ No timesteps with loss landscape data found")
            return []

        # Filter timesteps based on parameters
        if start_step is not None:
            all_steps = [s for s in all_steps if s >= start_step]
        if end_step is not None:
            all_steps = [s for s in all_steps if s <= end_step]

        # Apply step interval
        filtered_steps = all_steps[::step_interval]

        print(
            f"🎯 Processing {len(filtered_steps)} timesteps: {filtered_steps[:5]}{'...' if len(filtered_steps) > 5 else ''}"
        )
        print(f"📊 Plot settings: {figsize[0]}x{figsize[1]} @ {dpi}DPI")

        created_files = []
        start_time = time.time()

        for i, step in enumerate(filtered_steps):
            try:
                print(f"[{i+1}/{len(filtered_steps)}] Processing timestep {step}...", end=" ")

                # Load data
                data = self.visualizer.load_step_data(step)

                # Create main plot
                plot_filename = f"timestep_{step:04d}.png"
                plot_path = self.output_dir / "plots" / plot_filename

                self.create_enhanced_2d_plot(
                    data, step, str(plot_path), figsize, dpi, show_trajectory_numbers
                )
                created_files.append(str(plot_path))

                # Create thumbnail if requested
                if create_thumbnails:
                    thumb_path = self.output_dir / "thumbnails" / f"thumb_{plot_filename}"
                    self.create_thumbnail(str(plot_path), str(thumb_path))

                print("✅")

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

        elapsed = time.time() - start_time
        print(f"\n🎉 Completed! Created {len(created_files)} plots in {elapsed:.1f}s")
        print(f"📁 Plots saved to: {(self.output_dir / 'plots').absolute()}")
        if create_thumbnails:
            print(f"🖼️  Thumbnails saved to: {(self.output_dir / 'thumbnails').absolute()}")

        return created_files

    def create_index_html(self, created_files: List[str]):
        """Create an HTML index page to browse all plots."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Loss Landscape Visualization - {Path(self.hdf5_path).stem}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .plot-card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-card img {{ width: 100%; height: auto; border-radius: 4px; cursor: pointer; }}
        .plot-card h3 {{ margin: 10px 0 5px 0; color: #333; }}
        .plot-card p {{ color: #666; font-size: 14px; margin: 5px 0; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 15px 0; }}
        .stat {{ background: #e9ecef; padding: 10px; border-radius: 4px; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Loss Landscape Visualization</h1>
        <p><strong>Simulation:</strong> {Path(self.hdf5_path).stem}</p>
        <div class="stats">
            <div class="stat"><strong>{len(created_files)}</strong><br>Total Plots</div>
            <div class="stat"><strong>{len(self.visualizer.get_available_steps())}</strong><br>Available Timesteps</div>
            <div class="stat"><strong>{Path(self.hdf5_path).stat().st_size / 1024 / 1024:.1f} MB</strong><br>HDF5 File Size</div>
        </div>
    </div>
    
    <div class="gallery">
"""

        for plot_path in created_files:
            plot_file = Path(plot_path)
            timestep = int(plot_file.stem.split("_")[1])
            thumb_path = self.output_dir / "thumbnails" / f"thumb_{plot_file.name}"

            # Use thumbnail if available, otherwise use main plot
            img_src = (
                f"thumbnails/thumb_{plot_file.name}"
                if thumb_path.exists()
                else f"plots/{plot_file.name}"
            )

            html_content += f"""
        <div class="plot-card">
            <img src="{img_src}" alt="Timestep {timestep}" onclick="window.open('plots/{plot_file.name}', '_blank')">
            <h3>Timestep {timestep}</h3>
            <p>Click to view full resolution</p>
        </div>
"""

        html_content += """
    </div>
    
    <script>
        // Add keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                // Could implement next/prev navigation
            }
        });
    </script>
</body>
</html>
"""

        index_path = self.output_dir / "index.html"
        with open(index_path, "w") as f:
            f.write(html_content)

        print(f"🌐 Created HTML index: {index_path.absolute()}")
        return str(index_path)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Batch create 2D loss landscape visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("hdf5_file", help="Path to HDF5 file with simulation data")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="visualization_output",
        help="Output directory for plots (default: visualization_output)",
    )
    parser.add_argument("--start", type=int, help="First timestep to process")
    parser.add_argument("--end", type=int, help="Last timestep to process")
    parser.add_argument(
        "--step-interval", type=int, default=1, help="Process every N-th timestep (default: 1)"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[12, 9],
        help="Figure size in inches (default: 12 9)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved plots (default: 150)")
    parser.add_argument(
        "--no-thumbnails", action="store_true", help="Don't create thumbnail versions"
    )
    parser.add_argument(
        "--no-trajectory-numbers",
        action="store_true",
        help="Don't show iteration numbers on trajectory",
    )
    parser.add_argument("--no-html", action="store_true", help="Don't create HTML index page")
    parser.add_argument(
        "--list-steps", action="store_true", help="List available timesteps and exit"
    )

    args = parser.parse_args()

    # Initialize batch visualizer
    try:
        batch_viz = BatchVisualizer2D(args.hdf5_file, args.output_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
        return 1

    # List steps if requested
    if args.list_steps:
        steps = batch_viz.visualizer.get_available_steps()
        print(f"Available timesteps: {steps}")
        return 0

    # Process all timesteps
    created_files = batch_viz.process_all_timesteps(
        start_step=args.start,
        end_step=args.end,
        step_interval=args.step_interval,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        create_thumbnails=not args.no_thumbnails,
        show_trajectory_numbers=not args.no_trajectory_numbers,
    )

    if not created_files:
        print("❌ No plots were created")
        return 1

    # Create HTML index page
    if not args.no_html:
        batch_viz.create_index_html(created_files)

    print(f"\n✨ Success! Created {len(created_files)} visualizations")
    return 0


if __name__ == "__main__":
    exit(main())

