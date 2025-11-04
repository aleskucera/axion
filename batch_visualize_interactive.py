#!/usr/bin/env python3
"""
Batch Interactive Loss Landscape Visualization Script

This script processes an entire simulation HDF5 file and creates interactive Plotly
visualizations (2D and 3D) organized in an HTML gallery with thumbnails.

Usage:
    python batch_visualize_interactive.py <hdf5_file> [options]

Examples:
    # Basic batch processing - creates interactive gallery
    python batch_visualize_interactive.py simulation_data.h5

    # Custom output directory
    python batch_visualize_interactive.py simulation_data.h5 --output-dir interactive_gallery/

    # Process specific range with both 2D and 3D
    python batch_visualize_interactive.py simulation_data.h5 --start 10 --end 50 --modes 2d 3d

    # Process every 5th timestep
    python batch_visualize_interactive.py simulation_data.h5 --step-interval 5

    # Only 3D plots with custom size
    python batch_visualize_interactive.py simulation_data.h5 --modes 3d --width 1200 --height 900
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from axion.logging.hdf5_reader import HDF5Reader

# Import the visualizer class
from visualize_loss_landscape import LossLandscapeVisualizer

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import plot

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("❌ Plotly is required for interactive visualizations")
    sys.exit(1)


class BatchInteractiveVisualizer:
    """Batch processor for creating interactive Plotly loss landscape visualizations."""

    def __init__(self, hdf5_path: str, output_dir: str = "interactive_gallery"):
        """Initialize batch interactive visualizer.

        Args:
            hdf5_path: Path to HDF5 file with simulation data
            output_dir: Directory to save all plots and gallery
        """
        self.hdf5_path = hdf5_path
        self.output_dir = Path(output_dir)
        self.visualizer = LossLandscapeVisualizer(hdf5_path)

        # Create output directory structure
        self.setup_output_directory()

        # Store metadata for gallery
        self.plot_metadata = []

    def setup_output_directory(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "plots_2d").mkdir(exist_ok=True)
        (self.output_dir / "plots_3d").mkdir(exist_ok=True)
        (self.output_dir / "thumbnails").mkdir(exist_ok=True)
        (self.output_dir / "assets").mkdir(exist_ok=True)

        print(f"📁 Output directory: {self.output_dir.absolute()}")

    def create_interactive_2d_plot(
        self,
        data: Dict[str, np.ndarray],
        step: int,
        output_path: str,
        width: int = 900,
        height: int = 700,
    ) -> str:
        """Create interactive 2D Plotly visualization."""

        loss_grid = data["residual_norm_grid"]
        alphas = data["pca_alphas"]
        betas = data["pca_betas"]
        trajectory_2d = data["trajectory_2d_projected"]

        fig = go.Figure()

        # Handle wide dynamic range by using log scale for visualization
        loss_grid_clipped = np.clip(loss_grid.T, 1e-12, None)
        log_loss_grid = np.log10(loss_grid_clipped)

        # Calculate better contour levels for log scale
        log_min, log_max = log_loss_grid.min(), log_loss_grid.max()

        # Add contour with better formatting for wide dynamic ranges
        fig.add_trace(
            go.Contour(
                x=alphas,
                y=betas,
                z=log_loss_grid,
                colorscale="Plasma",
                contours=dict(
                    coloring="fill",
                    showlabels=True,
                    labelfont=dict(size=12, color="black"),
                    start=log_min,
                    end=log_max,
                    size=(log_max - log_min) / 15,
                ),
                colorbar=dict(
                    title="Residual Norm",
                    tickformat=".1e",
                    tickfont=dict(size=14, color="black"),
                    len=0.8,
                    thickness=20,
                    nticks=8,
                    tickvals=np.linspace(log_min, log_max, 8),
                    ticktext=[f"{10**val:.1e}" for val in np.linspace(log_min, log_max, 8)],
                ),
                name="Loss Landscape",
                hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Loss: %{text}<extra></extra>",
                text=[[f"{val:.2e}" for val in row] for row in loss_grid_clipped],
            )
        )

        # Add trajectory
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

        # Highlight start and end
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

        # Add metadata annotation
        metadata_text = ""
        if "pca_metadata" in data:
            metadata = data["pca_metadata"]
            metadata_text = (
                f"Grid: {int(metadata[0])}×{int(metadata[0])}, Iterations: {int(metadata[2])}"
            )

        fig.update_layout(
            title=dict(
                text=f"2D Loss Landscape - Timestep {step}<br><sub>{metadata_text}</sub>",
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
            width=width,
            height=height,
            font=dict(color="black", size=12),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        # Save as HTML
        fig.write_html(output_path, include_plotlyjs="cdn")
        return output_path

    def create_interactive_3d_plot(
        self,
        data: Dict[str, np.ndarray],
        step: int,
        output_path: str,
        width: int = 1000,
        height: int = 800,
    ) -> str:
        """Create interactive 3D Plotly visualization."""

        loss_grid = data["residual_norm_grid"]
        alphas = data["pca_alphas"]
        betas = data["pca_betas"]
        trajectory_2d = data["trajectory_2d_projected"]

        # Create meshgrid for 3D plot
        alpha_mesh, beta_mesh = np.meshgrid(alphas, betas)
        fig = go.Figure()

        # Prepare surface data
        loss_surface = np.clip(loss_grid.T, 1e-9, None)
        log_loss_surface = np.log10(loss_surface)

        # Add surface with contours
        fig.add_trace(
            go.Surface(
                x=alpha_mesh,
                y=beta_mesh,
                z=loss_surface,
                surfacecolor=log_loss_surface,
                colorscale="Plasma",
                name="Loss Surface",
                showscale=True,
                opacity=0.7,
                colorbar=dict(
                    title="Residual Norm",
                    tickformat=".1e",
                    tickfont=dict(size=14),
                    len=0.7,
                    thickness=20,
                    nticks=6,
                    tick0=log_loss_surface.min(),
                    dtick=(log_loss_surface.max() - log_loss_surface.min()) / 5,
                ),
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

        # Add trajectory
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator((alphas, betas), loss_grid)
        traj_loss = np.clip(interp(trajectory_2d), 1e-9, None)

        fig.add_trace(
            go.Scatter3d(
                x=trajectory_2d[:, 0],
                y=trajectory_2d[:, 1],
                z=traj_loss,
                mode="lines+markers",
                line=dict(color="black", width=4),
                marker=dict(color="white", size=4, line=dict(color="black", width=1)),
                name="Optimization Path",
                text=[f"Iteration {i}: {loss:.2e}" for i, loss in enumerate(traj_loss)],
                hovertemplate="%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

        # Add start and end markers
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

        # Add metadata to title
        metadata_text = ""
        if "pca_metadata" in data:
            metadata = data["pca_metadata"]
            metadata_text = (
                f"Grid: {int(metadata[0])}×{int(metadata[0])}, Iterations: {int(metadata[2])}"
            )

        fig.update_layout(
            title=dict(
                text=f"3D Loss Landscape - Timestep {step}<br><sub>{metadata_text}</sub>",
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
                    type="log",
                    tickformat=".1e",
                    title="Residual Norm",
                    tickfont=dict(size=14, color="black"),
                    gridcolor="lightgray",
                    showbackground=True,
                    backgroundcolor="white",
                    nticks=5,
                    exponentformat="e",
                ),
                bgcolor="white",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            template="plotly_white",
            width=width,
            height=height,
            font=dict(color="black", size=12),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        # Save as HTML
        fig.write_html(output_path, include_plotlyjs="cdn")
        return output_path

    def create_thumbnail_from_plotly(
        self, plot_path: str, thumbnail_path: str, width: int = 300, height: int = 225
    ):
        """Create thumbnail from Plotly HTML using screenshot (requires kaleido)."""
        try:
            import kaleido

            # Read the HTML and extract the plot
            with open(plot_path, "r") as f:
                html_content = f.read()

            # This is complex - for now, create a simple placeholder
            self.create_placeholder_thumbnail(thumbnail_path, width, height, Path(plot_path).stem)

        except ImportError:
            # Create placeholder thumbnail instead
            self.create_placeholder_thumbnail(thumbnail_path, width, height, Path(plot_path).stem)

    def create_placeholder_thumbnail(
        self, thumbnail_path: str, width: int, height: int, title: str
    ):
        """Create a simple SVG placeholder thumbnail."""
        timestep = title.split("_")[-1] if "timestep" in title else title
        mode = "3D" if "3d" in str(thumbnail_path) else "2D"

        svg_content = f"""
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
    <circle cx="{width//2}" cy="{height//2-20}" r="30" fill="#6c757d" opacity="0.3"/>
    <text x="{width//2}" y="{height//2-15}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#495057">{mode} Plot</text>
    <text x="{width//2}" y="{height//2+5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#6c757d">Timestep {timestep}</text>
    <text x="{width//2}" y="{height-15}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#adb5bd">Click to view</text>
</svg>
"""

        with open(thumbnail_path, "w") as f:
            f.write(svg_content)

    def process_all_timesteps(
        self,
        modes: List[str] = ["2d", "3d"],
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        step_interval: int = 1,
        width: int = 900,
        height: int = 700,
        create_thumbnails: bool = True,
    ) -> Dict[str, List[str]]:
        """Process all timesteps and create interactive plots.

        Args:
            modes: List of plot modes to create ('2d', '3d', or both)
            start_step: First timestep to process
            end_step: Last timestep to process
            step_interval: Process every N-th timestep
            width: Plot width in pixels
            height: Plot height in pixels
            create_thumbnails: Whether to create thumbnails

        Returns:
            Dictionary with lists of created files by mode
        """
        # Get available timesteps
        all_steps = self.visualizer.get_available_steps()

        if not all_steps:
            print("❌ No timesteps with loss landscape data found")
            return {}

        # Filter timesteps
        if start_step is not None:
            all_steps = [s for s in all_steps if s >= start_step]
        if end_step is not None:
            all_steps = [s for s in all_steps if s <= end_step]

        filtered_steps = all_steps[::step_interval]

        print(f"🎯 Processing {len(filtered_steps)} timesteps with modes: {modes}")
        print(f"📊 Plot settings: {width}x{height} pixels")

        created_files = {mode: [] for mode in modes}
        start_time = time.time()

        for i, step in enumerate(filtered_steps):
            try:
                print(f"[{i+1}/{len(filtered_steps)}] Processing timestep {step}...", end=" ")

                # Load data once per timestep
                data = self.visualizer.load_step_data(step)

                step_metadata = {
                    "timestep": step,
                    "files": {},
                    "metadata": (
                        data.get("pca_metadata", []).tolist() if "pca_metadata" in data else []
                    ),
                }

                # Create plots for each requested mode
                for mode in modes:
                    if mode == "2d":
                        plot_filename = f"timestep_{step:04d}_2d.html"
                        plot_path = self.output_dir / "plots_2d" / plot_filename

                        self.create_interactive_2d_plot(data, step, str(plot_path), width, height)
                        created_files["2d"].append(str(plot_path))
                        step_metadata["files"]["2d"] = f"plots_2d/{plot_filename}"

                        if create_thumbnails:
                            thumb_path = self.output_dir / "thumbnails" / f"thumb_{step:04d}_2d.svg"
                            self.create_placeholder_thumbnail(
                                str(thumb_path), 300, 225, f"timestep_{step:04d}_2d"
                            )

                    elif mode == "3d":
                        plot_filename = f"timestep_{step:04d}_3d.html"
                        plot_path = self.output_dir / "plots_3d" / plot_filename

                        self.create_interactive_3d_plot(data, step, str(plot_path), width, height)
                        created_files["3d"].append(str(plot_path))
                        step_metadata["files"]["3d"] = f"plots_3d/{plot_filename}"

                        if create_thumbnails:
                            thumb_path = self.output_dir / "thumbnails" / f"thumb_{step:04d}_3d.svg"
                            self.create_placeholder_thumbnail(
                                str(thumb_path), 300, 225, f"timestep_{step:04d}_3d"
                            )

                self.plot_metadata.append(step_metadata)
                print("✅")

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

        elapsed = time.time() - start_time
        total_plots = sum(len(files) for files in created_files.values())
        print(f"\n🎉 Completed! Created {total_plots} interactive plots in {elapsed:.1f}s")

        for mode, files in created_files.items():
            if files:
                print(
                    f"📁 {mode.upper()} plots: {len(files)} files in {(self.output_dir / f'plots_{mode}').absolute()}"
                )

        return created_files

    def create_interactive_gallery(self, created_files: Dict[str, List[str]]):
        """Create interactive HTML gallery with mode switching."""

        # Create gallery data
        gallery_data = {
            "simulation": Path(self.hdf5_path).stem,
            "timesteps": self.plot_metadata,
            "modes": list(created_files.keys()),
            "total_plots": sum(len(files) for files in created_files.values()),
            "hdf5_size_mb": round(Path(self.hdf5_path).stat().st_size / 1024 / 1024, 1),
        }

        # Save metadata as JSON
        with open(self.output_dir / "assets" / "gallery_data.json", "w") as f:
            json.dump(gallery_data, f, indent=2)

        # Create main gallery HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Loss Landscape Gallery - {gallery_data['simulation']}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .controls {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }}
        
        .mode-selector {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }}
        
        .mode-btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: #e9ecef;
            color: #495057;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .mode-btn.active {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }}
        
        .search-filter {{
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .search-input {{
            padding: 10px 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 16px;
            min-width: 200px;
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 25px;
        }}
        
        .plot-card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            opacity: 1;
        }}
        
        .plot-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 16px 48px rgba(0,0,0,0.15);
        }}
        
        .plot-card.hidden {{
            opacity: 0;
            transform: scale(0.8);
            pointer-events: none;
        }}
        
        .plot-thumbnail {{
            width: 100%;
            height: 200px;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }}
        
        .plot-thumbnail svg {{
            width: 100%;
            height: 100%;
        }}
        
        .plot-thumbnail::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 0%, rgba(102, 126, 234, 0.1) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .plot-thumbnail:hover::after {{
            opacity: 1;
        }}
        
        .plot-info {{
            padding: 20px;
        }}
        
        .plot-title {{
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }}
        
        .plot-meta {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        
        .plot-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .plot-btn {{
            flex: 1;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .plot-btn.primary {{
            background: #667eea;
            color: white;
        }}
        
        .plot-btn.primary:hover {{
            background: #5a67d8;
            transform: translateY(-1px);
        }}
        
        .plot-btn.secondary {{
            background: #e9ecef;
            color: #495057;
        }}
        
        .plot-btn.secondary:hover {{
            background: #dee2e6;
        }}
        
        .footer {{
            text-align: center;
            padding: 40px 20px;
            color: rgba(255,255,255,0.8);
        }}
        
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
        }}
        
        .modal-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90vw;
            height: 90vh;
            background: white;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .modal-header {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .modal-close {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #6c757d;
        }}
        
        .modal-close:hover {{
            color: #495057;
        }}
        
        .modal-frame {{
            width: 100%;
            height: calc(100% - 70px);
            border: none;
        }}
        
        @media (max-width: 768px) {{
            .gallery {{
                grid-template-columns: 1fr;
            }}
            
            .modal-content {{
                width: 95vw;
                height: 95vh;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 Interactive Loss Landscape Gallery</h1>
            <p><strong>Simulation:</strong> {gallery_data['simulation']}</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{gallery_data['total_plots']}</div>
                    <div>Total Plots</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(gallery_data['timesteps'])}</div>
                    <div>Timesteps</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{gallery_data['hdf5_size_mb']}</div>
                    <div>MB Data</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(gallery_data['modes'])}</div>
                    <div>Plot Types</div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <div class="mode-selector">
"""

        # Add mode buttons
        for i, mode in enumerate(gallery_data["modes"]):
            active_class = "active" if i == 0 else ""
            html_content += f'                <button class="mode-btn {active_class}" data-mode="{mode}">{mode.upper()} View</button>\n'

        html_content += (
            """
            </div>
            <div class="search-filter">
                <input type="text" class="search-input" placeholder="Search timesteps..." id="searchInput">
                <select class="search-input" id="sortSelect">
                    <option value="timestep-asc">Timestep (Low to High)</option>
                    <option value="timestep-desc">Timestep (High to Low)</option>
                </select>
            </div>
        </div>
        
        <div class="gallery" id="gallery">
            <!-- Gallery items will be populated by JavaScript -->
        </div>
    </div>
    
    <div class="footer">
        <p>Generated with Axion Loss Landscape Visualizer</p>
    </div>
    
    <!-- Modal for full-screen viewing -->
    <div class="modal" id="plotModal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">Plot Viewer</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <iframe class="modal-frame" id="modalFrame" src=""></iframe>
        </div>
    </div>

    <script>
        // Gallery data
        const galleryData = """
            + json.dumps(gallery_data, indent=8)
            + """;
        
        let currentMode = '"""
            + gallery_data["modes"][0]
            + """';
        
        // Initialize gallery
        document.addEventListener('DOMContentLoaded', function() {
            setupModeButtons();
            setupSearch();
            renderGallery();
        });
        
        function setupModeButtons() {
            const modeButtons = document.querySelectorAll('.mode-btn');
            modeButtons.forEach(btn => {
                btn.addEventListener('click', function() {
                    modeButtons.forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    currentMode = this.dataset.mode;
                    renderGallery();
                });
            });
        }
        
        function setupSearch() {
            const searchInput = document.getElementById('searchInput');
            const sortSelect = document.getElementById('sortSelect');
            
            searchInput.addEventListener('input', renderGallery);
            sortSelect.addEventListener('change', renderGallery);
        }
        
        function renderGallery() {
            const gallery = document.getElementById('gallery');
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const sortOrder = document.getElementById('sortSelect').value;
            
            // Filter and sort timesteps
            let filteredData = galleryData.timesteps.filter(item => {
                const hasMode = item.files[currentMode];
                const matchesSearch = item.timestep.toString().includes(searchTerm);
                return hasMode && matchesSearch;
            });
            
            // Sort data
            filteredData.sort((a, b) => {
                if (sortOrder === 'timestep-asc') return a.timestep - b.timestep;
                if (sortOrder === 'timestep-desc') return b.timestep - a.timestep;
                return 0;
            });
            
            // Generate HTML
            gallery.innerHTML = filteredData.map(item => {
                const thumbnailPath = `thumbnails/thumb_${item.timestep.toString().padStart(4, '0')}_${currentMode}.svg`;
                const plotPath = item.files[currentMode];
                const metadata = item.metadata.length > 0 ? 
                    `Grid: ${item.metadata[0]}×${item.metadata[0]}, Iterations: ${item.metadata[2]}` : 
                    'No metadata available';
                
                return `
                    <div class="plot-card">
                        <div class="plot-thumbnail" onclick="openModal('${plotPath}', 'Timestep ${item.timestep} - ${currentMode.toUpperCase()}')">
                            <object data="${thumbnailPath}" type="image/svg+xml">
                                <div style="padding: 20px; text-align: center; color: #6c757d;">
                                    <div style="font-size: 2em; margin-bottom: 10px;">📊</div>
                                    <div>${currentMode.toUpperCase()} Plot</div>
                                    <div>Timestep ${item.timestep}</div>
                                </div>
                            </object>
                        </div>
                        <div class="plot-info">
                            <div class="plot-title">Timestep ${item.timestep}</div>
                            <div class="plot-meta">${metadata}</div>
                            <div class="plot-actions">
                                <button class="plot-btn primary" onclick="openModal('${plotPath}', 'Timestep ${item.timestep} - ${currentMode.toUpperCase()}')">
                                    View Interactive
                                </button>
                                <button class="plot-btn secondary" onclick="window.open('${plotPath}', '_blank')">
                                    Open in New Tab
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Update visibility with animation
            const cards = gallery.querySelectorAll('.plot-card');
            cards.forEach((card, index) => {
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 50);
            });
        }
        
        function openModal(plotPath, title) {
            const modal = document.getElementById('plotModal');
            const frame = document.getElementById('modalFrame');
            const modalTitle = document.getElementById('modalTitle');
            
            modalTitle.textContent = title;
            frame.src = plotPath;
            modal.style.display = 'block';
            
            // Close on escape key
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') closeModal();
            });
        }
        
        function closeModal() {
            const modal = document.getElementById('plotModal');
            const frame = document.getElementById('modalFrame');
            
            modal.style.display = 'none';
            frame.src = '';
        }
        
        // Close modal when clicking outside
        document.getElementById('plotModal').addEventListener('click', function(e) {
            if (e.target === this) closeModal();
        });
    </script>
</body>
</html>
"""
        )

        # Save gallery HTML
        gallery_path = self.output_dir / "index.html"
        with open(gallery_path, "w") as f:
            f.write(html_content)

        print(f"🌐 Created interactive gallery: {gallery_path.absolute()}")
        return str(gallery_path)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Batch create interactive loss landscape visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("hdf5_file", help="Path to HDF5 file with simulation data")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="interactive_gallery",
        help="Output directory for plots (default: interactive_gallery)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["2d", "3d"],
        default=["2d", "3d"],
        help="Plot modes to create (default: 2d 3d)",
    )
    parser.add_argument("--start", type=int, help="First timestep to process")
    parser.add_argument("--end", type=int, help="Last timestep to process")
    parser.add_argument(
        "--step-interval", type=int, default=1, help="Process every N-th timestep (default: 1)"
    )
    parser.add_argument(
        "--width", type=int, default=900, help="Plot width in pixels (default: 900)"
    )
    parser.add_argument(
        "--height", type=int, default=700, help="Plot height in pixels (default: 700)"
    )
    parser.add_argument(
        "--no-thumbnails", action="store_true", help="Don't create thumbnail placeholders"
    )
    parser.add_argument(
        "--list-steps", action="store_true", help="List available timesteps and exit"
    )

    args = parser.parse_args()

    # Initialize batch visualizer
    try:
        batch_viz = BatchInteractiveVisualizer(args.hdf5_file, args.output_dir)
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
        modes=args.modes,
        start_step=args.start,
        end_step=args.end,
        step_interval=args.step_interval,
        width=args.width,
        height=args.height,
        create_thumbnails=not args.no_thumbnails,
    )

    if not any(created_files.values()):
        print("❌ No plots were created")
        return 1

    # Create interactive gallery
    batch_viz.create_interactive_gallery(created_files)

    total_plots = sum(len(files) for files in created_files.values())
    print(f"\n✨ Success! Created interactive gallery with {total_plots} visualizations")
    print(f"🌐 Open the gallery: {(Path(args.output_dir) / 'index.html').absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
