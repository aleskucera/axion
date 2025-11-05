#!/usr/bin/env python3
"""
Loss Landscape Visualization Script

This script reads loss landscape data stored in HDF5 files by the Axion engine
and creates various visualizations of the optimization landscape.

Usage:
    python visualize_loss_landscape.py <hdf5_file> [options]

Examples:
    # Basic 2D contour plot
    python visualize_loss_landscape.py simulation_data.h5 --step 0

    # Interactive plot with all steps
    python visualize_loss_landscape.py simulation_data.h5 --interactive

    # 3D surface plot
    python visualize_loss_landscape.py simulation_data.h5 --step 0 --mode 3d

    # Batch process all steps
    python visualize_loss_landscape.py simulation_data.h5 --batch
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add src to path to import HDF5Reader
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from axion.logging.hdf5_reader import HDF5Reader

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive plots will be disabled.")


class LossLandscapeVisualizer:
    """Main class for creating loss landscape visualizations."""

    def __init__(self, hdf5_path: str):
        """Initialize with path to HDF5 file."""
        self.hdf5_path = hdf5_path
        self.reader = None
        self._validate_file()

    def _validate_file(self):
        """Validate that the HDF5 file exists and contains required data."""
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Quick check for required groups using HDF5Reader
        with HDF5Reader(self.hdf5_path) as reader:
            groups = reader.list_groups()
            timestep_groups = [g for g in groups if g.startswith("timestep_")]
            if not timestep_groups:
                raise ValueError("No timestep data found in HDF5 file")

            # Check if any timestep has loss landscape data
            has_landscape_data = False
            for timestep in timestep_groups:
                try:
                    landscape_groups = reader.list_groups(timestep)
                    if "residual_norm_landscape_data" in landscape_groups:
                        has_landscape_data = True
                        break
                except:
                    continue

            if not has_landscape_data:
                raise ValueError("No loss landscape data found in any timestep")

    def get_available_steps(self) -> List[int]:
        """Get list of available simulation steps with loss landscape data."""
        steps = []
        with HDF5Reader(self.hdf5_path) as reader:
            groups = reader.list_groups()
            for group_name in groups:
                if group_name.startswith("timestep_"):
                    try:
                        # Check if this timestep has loss landscape data
                        subgroups = reader.list_groups(group_name)
                        if "residual_norm_landscape_data" in subgroups:
                            step_num = int(group_name.split("_")[1])
                            steps.append(step_num)
                    except (ValueError, KeyError):
                        continue
        return sorted(steps)

    def load_step_data(self, step: int) -> Dict[str, np.ndarray]:
        """Load loss landscape data for a specific simulation step."""
        with HDF5Reader(self.hdf5_path) as reader:
            step_path = f"timestep_{step:04d}/residual_norm_landscape_data"

            # Check if the timestep exists
            try:
                datasets = reader.list_datasets(step_path)
            except KeyError:
                raise KeyError(f"No loss landscape data found for timestep {step}")

            data = {}
            required_keys = [
                "residual_norm_grid",
                "pca_alphas",
                "pca_betas",
                "trajectory_2d_projected",
                "pca_metadata",
            ]

            for key in required_keys:
                if key in datasets:
                    data[key] = reader.get_dataset(f"{step_path}/{key}")
                else:
                    raise KeyError(f"Required dataset '{key}' not found for timestep {step}")

            # Optional datasets
            optional_keys = [
                "pca_v1",
                "pca_v2",
                "pca_singular_values",
                "pca_center_point",
                "optimization_trajectory",
            ]

            for key in optional_keys:
                if key in datasets:
                    data[key] = reader.get_dataset(f"{step_path}/{key}")

        return data

    def create_2d_contour_plot(
        self,
        data: Dict[str, np.ndarray],
        step: int,
        output_path: Optional[str] = None,
        enhanced: bool = True,
    ) -> str:
        """Create enhanced 2D contour plot of the loss landscape."""

        loss_grid = data["residual_norm_grid"]
        alphas = data["pca_alphas"]
        betas = data["pca_betas"]
        trajectory_2d = data["trajectory_2d_projected"]

        fig, ax = plt.subplots(figsize=(12, 9))

        # Clip small values to prevent log(0) errors
        loss_grid_clipped = np.clip(loss_grid, 1e-9, None)

        # Enhanced contour plot with more levels and better colormap
        levels = np.logspace(
            np.log10(loss_grid_clipped.min()), np.log10(loss_grid_clipped.max()), 25
        )

        contour = ax.contourf(
            alphas,
            betas,
            loss_grid_clipped.T,
            levels=levels,
            cmap="viridis",
            norm=plt.matplotlib.colors.LogNorm(),
        )

        # Add contour lines for better structure visibility
        contour_lines = ax.contour(
            alphas,
            betas,
            loss_grid_clipped.T,
            levels=levels[::3],
            colors="white",
            alpha=0.3,
            linewidths=0.5,
        )

        # Enhanced colorbar with better tick formatting
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Residual Norm", fontsize=12)

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

        if enhanced and len(trajectory_2d) > 1:
            # Plot trajectory with iteration numbers
            ax.plot(
                trajectory_2d[:, 0],
                trajectory_2d[:, 1],
                "w-",
                linewidth=2,
                alpha=0.8,
                label="Optimization Path",
            )

            # Mark every few iterations with numbers
            step_size = max(1, len(trajectory_2d) // 10)
            for i in range(0, len(trajectory_2d), step_size):
                ax.plot(
                    trajectory_2d[i, 0],
                    trajectory_2d[i, 1],
                    "wo",
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=1,
                )
                ax.annotate(
                    str(i),
                    (trajectory_2d[i, 0], trajectory_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )

            # Highlight start and end
            ax.plot(
                trajectory_2d[0, 0],
                trajectory_2d[0, 1],
                "go",
                markersize=10,
                label="Start",
                markeredgecolor="black",
            )
            ax.plot(
                trajectory_2d[-1, 0],
                trajectory_2d[-1, 1],
                "rx",
                markersize=12,
                markeredgewidth=3,
                label="Solution",
            )
        else:
            # Simple trajectory for basic mode
            ax.plot(
                trajectory_2d[:, 0],
                trajectory_2d[:, 1],
                "w-o",
                markersize=4,
                linewidth=1.5,
                label="Optimization Path",
            )

        # Enhanced styling
        ax.set_title(f"Loss Landscape - Simulation Step {step}", fontsize=14, weight="bold")
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.3)

        # Add metadata if available
        if "pca_metadata" in data:
            metadata = data["pca_metadata"]
            info_text = (
                f"Grid: {int(metadata[0])}×{int(metadata[0])}, Iterations: {int(metadata[2])}"
            )
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=10,
            )

        plt.tight_layout()

        # Save plot
        if output_path is None:
            output_path = f"loss_landscape_step_{step}.png"

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def create_interactive_plot(
        self, data: Dict[str, np.ndarray], step: int, output_path: Optional[str] = None
    ) -> str:
        """Create interactive Plotly visualization."""

        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots")

        loss_grid = data["residual_norm_grid"]
        alphas = data["pca_alphas"]
        betas = data["pca_betas"]
        trajectory_2d = data["trajectory_2d_projected"]

        # Create contour plot
        fig = go.Figure()

        # Handle wide dynamic range by using log scale for visualization
        loss_grid_clipped = np.clip(loss_grid.T, 1e-12, None)  # Avoid log(0)
        log_loss_grid = np.log10(loss_grid_clipped)

        # Calculate better contour levels for log scale
        log_min, log_max = log_loss_grid.min(), log_loss_grid.max()
        contour_levels = np.linspace(log_min, log_max, 20)

        # Add contour with better formatting for wide dynamic ranges
        fig.add_trace(
            go.Contour(
                x=alphas,
                y=betas,
                z=log_loss_grid,
                colorscale="Plasma",  # Better colorscale
                contours=dict(
                    coloring="fill",
                    showlabels=True,
                    labelfont=dict(size=12, color="black"),
                    start=log_min,
                    end=log_max,
                    size=(log_max - log_min) / 15,  # Better level spacing
                ),
                colorbar=dict(
                    title="Residual Norm",
                    tickformat=".1e",  # Scientific notation
                    tickfont=dict(size=14, color="black"),
                    len=0.8,
                    thickness=20,
                    nticks=8,  # Limit ticks for readability
                    # Convert log values back to actual values for display
                    tickvals=np.linspace(log_min, log_max, 8),
                    ticktext=[f"{10**val:.1e}" for val in np.linspace(log_min, log_max, 8)],
                ),
                name="Loss Landscape",
                # Custom hover template to show actual values (not log)
                hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Loss: %{text}<extra></extra>",
                text=[
                    [f"{val:.2e}" for val in row] for row in loss_grid_clipped
                ],  # Pre-format the values
            )
        )

        # Add trajectory with publication-quality styling
        fig.add_trace(
            go.Scatter(
                x=trajectory_2d[:, 0],
                y=trajectory_2d[:, 1],
                mode="lines+markers",
                line=dict(color="black", width=3),
                marker=dict(color="white", size=8, line=dict(color="black", width=2)),
                name="Optimization Path",
                text=[f"Iteration {i}" for i in range(len(trajectory_2d))],
                hovertemplate="Iteration %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

        # Highlight start and end with publication styling
        fig.add_trace(
            go.Scatter(
                x=[trajectory_2d[0, 0]],
                y=[trajectory_2d[0, 1]],
                mode="markers",
                marker=dict(
                    color="green", size=14, symbol="circle", line=dict(color="black", width=2)
                ),
                name="Start",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[trajectory_2d[-1, 0]],
                y=[trajectory_2d[-1, 1]],
                mode="markers",
                marker=dict(color="red", size=14, symbol="x", line=dict(color="black", width=2)),
                name="Solution",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Loss Landscape Visualization - Timestep {step}",
                font=dict(size=18, color="black"),
            ),
            xaxis=dict(
                title="Principal Component 1",
                tickfont=dict(size=14, color="black"),
                gridcolor="lightgray",
                showgrid=True,
            ),
            yaxis=dict(
                title="Principal Component 2",
                tickfont=dict(size=14, color="black"),
                gridcolor="lightgray",
                showgrid=True,
            ),
            template="plotly_white",
            width=900,
            height=700,
            font=dict(color="black", size=12),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        if output_path is None:
            output_path = f"loss_landscape_step_{step}_interactive.html"

        fig.write_html(output_path)
        return output_path

    def create_3d_surface_plot(
        self, data: Dict[str, np.ndarray], step: int, output_path: Optional[str] = None
    ) -> str:
        """Create 3D surface plot of the loss landscape."""

        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D plots")

        loss_grid = data["residual_norm_grid"]
        alphas = data["pca_alphas"]
        betas = data["pca_betas"]
        trajectory_2d = data["trajectory_2d_projected"]

        # Create meshgrid for 3D plot
        alpha_mesh, beta_mesh = np.meshgrid(alphas, betas)

        fig = go.Figure()

        # Prepare surface data with actual values (not log)
        loss_surface = np.clip(loss_grid.T, 1e-9, None)

        # Ensure surface shows color variation by using log-scaled surface
        log_loss_surface = np.log10(loss_surface)

        # Add surface with proper colorbar formatting and transparency
        fig.add_trace(
            go.Surface(
                x=alpha_mesh,
                y=beta_mesh,
                z=loss_surface,
                surfacecolor=log_loss_surface,  # Use log scale for colors only
                colorscale="Plasma",  # Better colorscale with more variation
                name="Loss Surface",
                showscale=True,
                opacity=0.7,  # Make surface semi-transparent to see trajectory
                colorbar=dict(
                    title="Residual Norm",
                    tickformat=".1e",  # Scientific notation with 1 decimal
                    tickfont=dict(size=14),
                    len=0.7,
                    thickness=20,
                    nticks=6,  # Limit number of ticks
                    # Set specific tick values to avoid crowding
                    tick0=log_loss_surface.min(),
                    dtick=(log_loss_surface.max() - log_loss_surface.min()) / 5,
                ),
                # Add contour lines to the surface
                contours=dict(
                    x=dict(
                        show=True,
                        start=alphas.min(),
                        end=alphas.max(),
                        size=(alphas.max() - alphas.min()) / 8,
                        color="rgba(70,70,70,0.6)",
                        width=1,
                    ),
                    y=dict(
                        show=True,
                        start=betas.min(),
                        end=betas.max(),
                        size=(betas.max() - betas.min()) / 8,
                        color="rgba(70,70,70,0.6)",
                        width=1,
                    ),
                    z=dict(
                        show=True,
                        start=loss_surface.min(),
                        end=loss_surface.max(),
                        project=dict(z=True),
                        color="rgba(70,70,70,0.5)",
                        width=2,
                    ),
                ),
                hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Loss: %{z:.2e}<extra></extra>",
            )
        )

        # Add trajectory projected onto surface
        # Interpolate loss values for trajectory points
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator((alphas, betas), loss_grid)
        traj_loss = np.clip(interp(trajectory_2d), 1e-9, None)

        # Enhanced trajectory visualization with smaller, cleaner markers
        fig.add_trace(
            go.Scatter3d(
                x=trajectory_2d[:, 0],
                y=trajectory_2d[:, 1],
                z=traj_loss,
                mode="lines+markers",
                line=dict(color="black", width=4),  # Thinner line
                marker=dict(
                    color="white",
                    size=4,  # Much smaller markers
                    line=dict(color="black", width=1),  # Thin black outline
                ),
                name="Optimization Path",
                text=[f"Iteration {i}: {loss:.2e}" for i, loss in enumerate(traj_loss)],
                hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

        # Add start and end markers for clarity (smaller)
        fig.add_trace(
            go.Scatter3d(
                x=[trajectory_2d[0, 0]],
                y=[trajectory_2d[0, 1]],
                z=[traj_loss[0]],
                mode="markers",
                marker=dict(
                    color="green", size=8, symbol="circle", line=dict(color="black", width=1)
                ),
                name="Start",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[trajectory_2d[-1, 0]],
                y=[trajectory_2d[-1, 1]],
                z=[traj_loss[-1]],
                mode="markers",
                marker=dict(color="red", size=8, symbol="x", line=dict(color="black", width=1)),
                name="Solution",
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Loss Landscape Visualization - Timestep {step}",
                font=dict(size=18, color="black"),
            ),
            scene=dict(
                xaxis=dict(
                    title="Principal Component 1",
                    tickfont=dict(size=14, color="black"),
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                ),
                yaxis=dict(
                    title="Principal Component 2",
                    tickfont=dict(size=14, color="black"),
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                ),
                zaxis=dict(
                    type="log",  # Use log scale for z-axis
                    tickformat=".1e",  # Scientific notation with 1 decimal
                    title="Residual Norm",
                    tickfont=dict(size=14, color="black"),
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                    nticks=5,  # Limit number of ticks to avoid crowding
                    exponentformat="e",  # Force scientific notation format
                ),
                bgcolor="white",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Better default viewing angle
            ),
            template="plotly_white",  # Clean white background
            width=1000,
            height=800,
            font=dict(color="black", size=12),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        if output_path is None:
            output_path = f"loss_landscape_step_{step}_3d.html"

        fig.write_html(output_path)
        return output_path


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize loss landscapes from Axion simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("hdf5_file", help="Path to HDF5 file with simulation data")
    parser.add_argument("--step", type=int, help="Specific simulation step to visualize")
    parser.add_argument(
        "--mode",
        choices=["2d", "3d", "interactive"],
        default="2d",
        help="Visualization mode (default: 2d)",
    )
    parser.add_argument("--output", "-o", help="Output file path (auto-generated if not specified)")
    parser.add_argument("--batch", action="store_true", help="Process all available steps")
    parser.add_argument(
        "--enhanced", action="store_true", default=True, help="Use enhanced styling (default: True)"
    )
    parser.add_argument(
        "--list-steps", action="store_true", help="List available simulation steps and exit"
    )

    args = parser.parse_args()

    # Initialize visualizer
    try:
        viz = LossLandscapeVisualizer(args.hdf5_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    # List steps if requested
    if args.list_steps:
        steps = viz.get_available_steps()
        print(f"Available simulation steps: {steps}")
        return 0

    steps_to_process = []
    if args.batch:
        steps_to_process = viz.get_available_steps()
        if not steps_to_process:
            print("No loss landscape data found in HDF5 file")
            return 1
        print(f"Processing {len(steps_to_process)} steps: {steps_to_process}")
    else:
        if args.step is None:
            available_steps = viz.get_available_steps()
            if not available_steps:
                print("No loss landscape data found in HDF5 file")
                return 1
            args.step = available_steps[0]
            print(f"No step specified, using first available: {args.step}")
        steps_to_process = [args.step]

    # Process each step
    created_files = []
    for step in steps_to_process:
        try:
            print(f"Processing step {step}...")
            data = viz.load_step_data(step)

            output_path = args.output
            if args.batch and output_path:
                # Modify output path for batch processing
                path = Path(output_path)
                output_path = str(path.with_stem(f"{path.stem}_step_{step}"))

            if args.mode == "2d":
                created_file = viz.create_2d_contour_plot(data, step, output_path, args.enhanced)
            elif args.mode == "interactive":
                created_file = viz.create_interactive_plot(data, step, output_path)
            elif args.mode == "3d":
                created_file = viz.create_3d_surface_plot(data, step, output_path)

            created_files.append(created_file)
            print(f"✓ Created: {created_file}")

        except Exception as e:
            print(f"Error processing step {step}: {e}")
            if not args.batch:
                return 1

    print(f"\nSuccessfully created {len(created_files)} visualization(s)")
    return 0


if __name__ == "__main__":
    exit(main())
