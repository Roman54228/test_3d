#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ToF Viewer - Визуализатор данных Time-of-Flight камеры

Интерактивное приложение для просмотра IR изображений и данных глубины
с возможностью преобразования пиксельных координат в мировые.

Использование:
    python ir_click_viewer_simple.py

Управление:
    - Левая кнопка мыши: выбрать точку на изображении
    - Клавиша 'q': выход из приложения

Возможности:
    - Визуализация IR и depth данных в реальном времени
    - Интерактивный выбор точек мышью
    - Преобразование координат: пиксель → мир (2D/3D)
    - Отображение intrinsics камеры
    - Настраиваемые параметры ToF камеры

Автор: Refactored version
Дата: 2025
"""

import os
import sys
import locale
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyvidu as vidu
except ImportError:
    print("pyvidu module not found. Install the SDK and configure Python path.")
    sys.exit(1)

# Make project root importable to reach coordinate_transformer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from coordinate_transformer import CoordinateTransformer


class Config:
    """Configuration constants for the ToF viewer."""
    # Display settings
    DISPLAY_SIZE = (640, 480)
    WINDOW_NAME = "ToF Viewer"
    
    # Distance settings
    DEFAULT_DISTANCE_M = 1.49
    DEFAULT_TOF_RANGE_M = 7.5
    
    # Visualization settings
    DEPTH_SCALE_ALPHA = 2000.0 / 60000.0
    RAW_DEPTH_MULTIPLIER = 2
    MAX_DEPTH_VALUE = 65535.0
    
    # ToF camera parameters
    TOF_PARAMS = {
        "ToF::StreamFps": 5,
        "ToF::Distance": 7.5,
        "ToF::Exposure": 5.6,
        "ToF::DepthMedianBlur": 0,
        "ToF::DepthFlyingPixelRemoval": 1,
        "ToF::Threshold": 100,
        "ToF::Gain": 2,
        "ToF::AutoExposure": 0,
        "ToF::DepthSmoothStrength": 0,
        "ToF::DepthCompletion": 0,
    }
    
    # Text rendering settings
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.5
    TEXT_THICKNESS = 1
    TEXT_PADDING = 3
    TEXT_LINE_HEIGHT = 18
    TEXT_COLOR = (0, 255, 0)
    TEXT_BG_COLOR = (0, 0, 0)
    INTRINSICS_COLOR = (0, 255, 255)
    
    # Marker settings
    MARKER_COLOR = (0, 255, 0)
    MARKER_SIZE = 12
    MARKER_THICKNESS = 2


def setup_windows_encoding():
    """Configure Windows console for UTF-8 encoding."""
    if not sys.platform.startswith('win'):
        return
        
    os.system('chcp 65001 > nul')
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except Exception:
        pass


# Initialize encoding at module level
setup_windows_encoding()


class ViewerState:
    """Maintains the state of the ToF viewer application."""
    
    def __init__(self, display_size: Tuple[int, int] = Config.DISPLAY_SIZE) -> None:
        self.distance_range_m = Config.DEFAULT_TOF_RANGE_M
        self.display_size = display_size
        self.source_size: Optional[Tuple[int, int]] = None
        self.depth_frame: Optional[np.ndarray] = None
        self.click_position: Optional[Tuple[int, int]] = None
        self.world_2d: Optional[Tuple[float, float]] = None
        self.world_3d: Optional[Tuple[float, float, float]] = None
        self.intrinsics_text: Optional[Tuple[str, str, str]] = None
        self.transformer = CoordinateTransformer()

    def update_source_size(self, frame: np.ndarray) -> None:
        """Update source frame dimensions."""
        if frame is not None:
            height, width = frame.shape[:2]
            self.source_size = (width, height)

    def map_display_to_source(self, x: int, y: int) -> Tuple[int, int]:
        """Map display coordinates to source frame coordinates."""
        if not self.source_size:
            return x, y
            
        src_width, src_height = self.source_size
        disp_width, disp_height = self.display_size
        
        x_src = int(round(x * (src_width / float(disp_width))))
        y_src = int(round(y * (src_height / float(disp_height))))
        
        # Clamp to valid range
        x_src = max(0, min(src_width - 1, x_src))
        y_src = max(0, min(src_height - 1, y_src))
        
        return x_src, y_src
    
    def calculate_depth_distance(self, raw_depth: float) -> float:
        """Convert raw depth value to distance in meters."""
        if raw_depth <= 0:
            return 0.0
        raw_scaled = raw_depth * Config.RAW_DEPTH_MULTIPLIER
        return raw_scaled * (self.distance_range_m / Config.MAX_DEPTH_VALUE)


def extract_camera_intrinsics(stream, state: ViewerState) -> bool:
    """Extract camera intrinsic parameters and update state."""
    intrinsics = vidu.intrinsics()
    extrinsics = vidu.extrinsics()
    
    if not stream.getCamPara(intrinsics, extrinsics):
        return False
    
    # Extract all non-private, non-callable attributes
    attrs = {
        name.lower(): getattr(intrinsics, name) 
        for name in dir(intrinsics)
        if not name.startswith('__') and not callable(getattr(intrinsics, name))
    }
    
    # Try to get focal lengths and principal point
    fx = attrs.get('fx', attrs.get('f_x'))
    fy = attrs.get('fy', attrs.get('f_y'))
    cx = attrs.get('cx', attrs.get('c_x'))
    cy = attrs.get('cy', attrs.get('c_y'))
    
    # Fallback to intrinsic matrix K if individual parameters not available
    if (fx is None or fy is None or cx is None or cy is None) and hasattr(intrinsics, 'K'):
        K_matrix = np.array(getattr(intrinsics, 'K'), dtype=float).reshape(3, 3)
        fx, fy, cx, cy = K_matrix[0, 0], K_matrix[1, 1], K_matrix[0, 2], K_matrix[1, 2]
    
    if fx is None or fy is None or cx is None or cy is None:
        return False
    
    # Convert to float
    fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
    
    # Set camera matrix
    state.transformer.K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Extract distortion coefficients
    k1 = float(attrs.get('k1', 0.0))
    k2 = float(attrs.get('k2', 0.0))
    p1 = float(attrs.get('p1', 0.0))
    p2 = float(attrs.get('p2', 0.0))
    k3 = float(attrs.get('k3', 0.0))
    
    state.transformer.dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    
    # Store formatted text for display
    state.intrinsics_text = (
        f"[{fx:.3f}, 0.000, {cx:.3f}]",
        f"[0.000, {fy:.3f}, {cy:.3f}]",
        "[0.000, 0.000, 1.000]",
    )
    
    return True


def handle_mouse_click(event: int, x: int, y: int, flags: int, param) -> None:
    """Handle mouse click events and compute world coordinates."""
    state: ViewerState = param
    
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    
    # Store click position in display coordinates
    state.click_position = (x, y)
    
    # Map to source frame coordinates
    x_src, y_src = state.map_display_to_source(x, y)
    
    # Calculate 2D world coordinates (projection on Z=0 plane)
    world_x_2d, world_y_2d = state.transformer.pixel_to_world(x_src, y_src, None)
    state.world_2d = (float(world_x_2d), float(world_y_2d))
    
    # Calculate 3D world coordinates using fixed distance
    depth_mm = Config.DEFAULT_DISTANCE_M
    world_x, world_y, world_z = state.transformer.pixel_to_floor_3d(x_src, y_src, depth_mm)
    state.world_3d = (float(world_x), float(world_y), float(world_z))
    
    # Optional: Read actual depth from depth frame
    if state.depth_frame is not None:
        if 0 <= y_src < state.depth_frame.shape[0] and 0 <= x_src < state.depth_frame.shape[1]:
            raw_depth = float(state.depth_frame[y_src, x_src]) * Config.RAW_DEPTH_MULTIPLIER
            if raw_depth > 0:
                print(f"Raw depth: {raw_depth:.1f}, Range: {state.distance_range_m:.2f}m")
    
    # Log the click information
    depth_str = f"{depth_mm:.1f} mm"
    world_2d_str = f"({state.world_2d[0]:.2f}, {state.world_2d[1]:.2f})"
    world_3d_str = f"({state.world_3d[0]:.2f}, {state.world_3d[1]:.2f}, {state.world_3d[2]:.2f})"
    
    print(f"CLICK display=({x},{y}) source=({x_src},{y_src}) "
          f"depth={depth_str} world2D={world_2d_str} world3D={world_3d_str}")


def draw_text_with_background(image: np.ndarray, 
                              position: Tuple[int, int], 
                              text: str, 
                              color: Tuple[int, int, int] = Config.TEXT_COLOR) -> None:
    """Draw text with a background rectangle for better visibility."""
    text_size, _ = cv2.getTextSize(
        text, 
        Config.TEXT_FONT, 
        Config.TEXT_SCALE, 
        Config.TEXT_THICKNESS
    )
    text_width, text_height = text_size
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - Config.TEXT_PADDING, y - text_height - Config.TEXT_PADDING),
        (x + text_width + Config.TEXT_PADDING, y + Config.TEXT_PADDING),
        Config.TEXT_BG_COLOR,
        -1
    )
    
    # Draw text
    cv2.putText(
        image, 
        text, 
        (x, y), 
        Config.TEXT_FONT, 
        Config.TEXT_SCALE, 
        color, 
        Config.TEXT_THICKNESS, 
        cv2.LINE_AA
    )


def find_tof_stream(device) -> Optional[int]:
    """Find the ToF stream index from available device streams."""
    num_streams = device.getStreamNum()
    
    for stream_idx in range(num_streams):
        with vidu.PDstream(device, stream_idx) as stream:
            if not stream.init():
                continue
            
            stream_name = stream.getStreamName().upper()
            if "TOF" in stream_name:
                return stream_idx
    
    return None


def configure_tof_stream(stream) -> None:
    """Configure ToF stream parameters."""
    try:
        for param_name, param_value in Config.TOF_PARAMS.items():
            stream.set(param_name, param_value)
    except Exception as e:
        print(f"Warning: Failed to set some ToF parameters: {e}")


def process_frames(ir_frame: np.ndarray, depth_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Process and flip frames."""
    ir_flipped = cv2.flip(ir_frame, -1)
    depth_flipped = cv2.flip(depth_frame, -1)
    return ir_flipped, depth_flipped


def create_visualization(ir_frame: np.ndarray, state: ViewerState) -> np.ndarray:
    """Create visualization from IR frame."""
    # Convert to 8-bit and colorize
    vis = cv2.convertScaleAbs(ir_frame, alpha=Config.DEPTH_SCALE_ALPHA)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # Resize to display size
    vis = cv2.resize(vis, state.display_size)
    
    return vis


def draw_intrinsics_overlay(vis: np.ndarray, state: ViewerState) -> None:
    """Draw camera intrinsics on visualization."""
    if not state.intrinsics_text:
        return
    
    for idx, row_text in enumerate(state.intrinsics_text):
        y_pos = 20 + idx * Config.TEXT_LINE_HEIGHT
        draw_text_with_background(vis, (10, y_pos), row_text, Config.INTRINSICS_COLOR)


def draw_click_info_overlay(vis: np.ndarray, state: ViewerState) -> None:
    """Draw click position and world coordinates on visualization."""
    if not state.click_position:
        return
    
    x, y = state.click_position
    
    # Draw crosshair marker
    cv2.drawMarker(
        vis, 
        (x, y), 
        Config.MARKER_COLOR, 
        cv2.MARKER_CROSS, 
        Config.MARKER_SIZE, 
        Config.MARKER_THICKNESS
    )
    
    # Build information lines
    info_lines = [f"px=({x},{y})"]
    
    if state.world_2d:
        info_lines.append(f"world2D=({state.world_2d[0]:.2f},{state.world_2d[1]:.2f})")
    
    if state.world_3d:
        x_3d, y_3d, z_3d = state.world_3d
        info_lines.append(f"world3D=({x_3d:.2f},{y_3d:.2f},{z_3d:.2f})")
    
    # Draw info text near cursor
    text_y = max(20, y - 20)
    for idx, text in enumerate(info_lines):
        text_pos_y = text_y + idx * Config.TEXT_LINE_HEIGHT
        draw_text_with_background(vis, (x + 10, text_pos_y), text)


def run_viewer_loop(tof_stream, state: ViewerState) -> None:
    """Main viewer loop for processing and displaying frames."""
    print("Press 'q' to quit")
    
    while True:
        # Get frames from ToF stream
        images = tof_stream.getPyImage()
        
        # Check if valid frames received
        if not images or len(images) < 2 or images[1] is None or images[1].size == 0:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Process frames
        ir_frame = images[1]
        depth_frame = images[0]
        ir_processed, depth_processed = process_frames(ir_frame, depth_frame)
        
        # Update state
        state.depth_frame = depth_processed
        state.update_source_size(depth_processed)
        
        # Create visualization
        vis = create_visualization(ir_processed, state)
        
        # Draw overlays
        draw_intrinsics_overlay(vis, state)
        draw_click_info_overlay(vis, state)
        
        # Display
        cv2.imshow(Config.WINDOW_NAME, vis)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    """Main entry point for the ToF viewer application."""
    # Initialize device
    device = vidu.PDdevice()
    if not device.init():
        print("Error: Device initialization failed")
        sys.exit(1)
    
    # Find ToF stream
    tof_stream_idx = find_tof_stream(device)
    if tof_stream_idx is None:
        print("Error: No ToF stream found")
        sys.exit(1)
    
    # Initialize application state
    state = ViewerState()
    
    # Setup OpenCV window
    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(Config.WINDOW_NAME, handle_mouse_click, state)
    
    # Open and configure ToF stream
    with vidu.PDstream(device, tof_stream_idx) as tof_stream:
        if not tof_stream.init():
            print("Error: ToF stream initialization failed")
            sys.exit(1)
        
        # Configure stream parameters
        configure_tof_stream(tof_stream)
        
        # Extract camera intrinsics
        if not extract_camera_intrinsics(tof_stream, state):
            print("Warning: Could not extract camera intrinsics")
        
        # Run main viewer loop
        run_viewer_loop(tof_stream, state)
    
    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

