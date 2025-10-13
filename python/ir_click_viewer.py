#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import locale
import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyvidu as vidu
except ImportError:
    print("pyvidu module not found. Ensure SDK is installed and Python path is configured.")
    sys.exit(1)

# Ensure UTF-8 output on Windows console
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except Exception:
        pass

# Make project root importable to reach coordinate_transformer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from coordinate_transformer import CoordinateTransformer


class IRCursorAnnotator:
    def __init__(self) -> None:
        self.transformer = CoordinateTransformer()
        self.last_click_px: Optional[Tuple[int, int]] = None
        self.last_click_world2d: Optional[Tuple[float, float]] = None
        self.last_click_world3d: Optional[Tuple[float, float, float]] = None
        self.window_name = "RGB Viewer (click to annotate)"
        self.k_matrix_text: Optional[Tuple[str, str, str]] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.src_size: Optional[Tuple[int, int]] = None  # (w,h) of original RGB/depth
        self.disp_size: Tuple[int, int] = (640, 480)     # (w,h) for display

    def _process_rgb_image(self, image: np.ndarray) -> np.ndarray:
        """Resize RGB frame to 640x480 for display."""
        try:
            return cv2.resize(image, self.disp_size, interpolation=cv2.INTER_LINEAR)
        except Exception:
            return image

    def set_source_size_from_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        self.src_size = (w, h)

    def map_display_to_source(self, x: int, y: int) -> Tuple[int, int]:
        if self.src_size is None:
            return x, y
        src_w, src_h = self.src_size
        disp_w, disp_h = self.disp_size
        xs = int(round(x * (src_w / float(disp_w))))
        ys = int(round(y * (src_h / float(disp_h))))
        xs = max(0, min(src_w - 1, xs))
        ys = max(0, min(src_h - 1, ys))
        return xs, ys

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click_px = (x, y)
            xs, ys = self.map_display_to_source(x, y)
            # 2D world at Z=0 uses pixel coords; pass source-res coordinates
            wx2d, wy2d = self.transformer.pixel_to_world(xs, ys, None)
            self.last_click_world2d = (float(wx2d) * 2.0, float(wy2d) * 2.0)

            # 3D world using depth if available (depth in source resolution)
            z_mm = None
            if self.latest_depth is not None:
                if 0 <= ys < self.latest_depth.shape[0] and 0 <= xs < self.latest_depth.shape[1]:
                    raw = float(self.latest_depth[ys, xs])
                    z_mm = raw * 1000.0 if raw > 0 and raw < 10.0 else raw
            if z_mm is not None and z_mm > 0:
                z_mm = z_mm * (65535 / 7.5)
                X, Y, Z = self.transformer.pixel_to_world_3d(xs, ys, z_mm)
                self.last_click_world3d = (float(X) * 2.0, float(Y) * 2.0, float(Z))
            else:
                self.last_click_world3d = None

            # Console log: write px, depth, and world coordinates
            try:
                depth_str = f"{z_mm:.1f} mm" if (z_mm is not None and z_mm > 0) else "None"
                world2d_str = (
                    f"({self.last_click_world2d[0]:.2f}, {self.last_click_world2d[1]:.2f})"
                    if self.last_click_world2d is not None else "None"
                )
                if self.last_click_world3d is not None:
                    Xd, Yd, Zd = self.last_click_world3d
                    world3d_str = f"({Xd:.2f}, {Yd:.2f}, {Zd:.2f})"
                else:
                    world3d_str = "None"
                print(
                    f"CLICK display_px=({x},{y}) src_px=({xs},{ys}) depth={depth_str} "
                    f"world2D={world2d_str} world3D={world3d_str}"
                )
            except Exception:
                pass

    def overlay_annotation(self, frame: np.ndarray) -> np.ndarray:
        # Overlay intrinsics K if available
        if self.k_matrix_text is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            x0, y0 = 10, 20
            for idx, row_text in enumerate(self.k_matrix_text):
                y = y0 + idx * 18
                (tw, th), _ = cv2.getTextSize(row_text, font, scale, thickness)
                cv2.rectangle(frame, (x0 - 3, y - th - 3), (x0 + tw + 3, y + 3), (0, 0, 0), -1)
                cv2.putText(frame, row_text, (x0, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

        if self.last_click_px is None:
            return frame

        x, y = self.last_click_px
        wx2d, wy2d = self.last_click_world2d if self.last_click_world2d is not None else (None, None)
        world3d = self.last_click_world3d

        # Draw a small crosshair on display image
        color = (0, 255, 0) if frame.ndim == 3 else 255
        cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        # Compose text
        text_lines = [
            f"px=({x},{y})"
        ]
        if wx2d is not None and wy2d is not None:
            text_lines.append(f"world2D=({wx2d:.2f},{wy2d:.2f})")
        if self.latest_depth is not None:
            xs, ys = self.map_display_to_source(x, y)
            raw = float(self.latest_depth[ys, xs]) if 0 <= ys < self.latest_depth.shape[0] and 0 <= xs < self.latest_depth.shape[1] else float('nan')
            raw = raw #* (7.5 / 65535)
            depth_maybe_m = raw > 0 and raw < 10.0
            if depth_maybe_m:
                text_lines.append(f"depth~m={raw:.3f}")
            else:
                text_lines.append(f"depth~mm={raw:.1f}")
           
        if world3d is not None:
            X, Y, Z = world3d
            text_lines.append(f"world3D=({X:.2f},{Y:.2f},{Z:.2f})")

        # Put text with background for readability
        def put_label(img: np.ndarray, origin: Tuple[int, int], text: str) -> None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x0, y0 = origin
            pad = 3
            cv2.rectangle(img, (x0 - pad, y0 - th - pad), (x0 + tw + pad, y0 + pad), (0, 0, 0), -1)
            cv2.putText(img, text, (x0, y0), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

        y1 = max(20, y - 20)
        for idx, tline in enumerate(text_lines):
            put_label(frame, (x + 10, y1 + idx * 18), tline)
        return frame


def main(use_tof: bool = False) -> None:
    print("Initializing device...")
    intrinsic = vidu.intrinsics()
    extrinsic = vidu.extrinsics()

    device = vidu.PDdevice()
    if not device.init():
        print("Device initialization failed. Connect camera and install drivers.")
        sys.exit(1)

    stream_count = device.getStreamNum()
    print(f"Detected {stream_count} streams")

    annotator = IRCursorAnnotator()
    annotator.window_name = "ToF Viewer (click to annotate)" if use_tof else "RGB Viewer (click to annotate)"
    cv2.namedWindow(annotator.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(annotator.window_name, annotator.on_mouse)

    # Select streams
    rgb_index = None
    rgb_name = None
    tof_index = None
    for i in range(stream_count):
        with vidu.PDstream(device, i) as s:
            if not s.init():
                continue
            name = s.getStreamName()
            u = name.upper()
            if rgb_index is None and ("RGB" in u or "COLOR" in u):
                rgb_index = i
                rgb_name = name
            if tof_index is None and ("TOF" in u):
                tof_index = i
            if rgb_index is not None and tof_index is not None:
                break

    if use_tof:
        if tof_index is None:
            print("No ToF stream found. Use without --tof or connect a ToF device.")
            sys.exit(1)
        print(f"Using ToF stream {tof_index} as main display")
    else:
        if rgb_index is None:
            print("No RGB/Color stream found. Cannot proceed.")
            sys.exit(1)
        print(f"Using RGB stream {rgb_index}: {rgb_name}")
        if tof_index is not None:
            print(f"Using ToF stream {tof_index} for depth")

    # Open streams
    # Open streams depending on mode
    if use_tof:
        rgb_stream = None
        with vidu.PDstream(device, tof_index) as tof_stream:
            if not tof_stream.init():
                print("ToF stream init failed")
                sys.exit(1)

            # Configure ToF camera similarly to camera_handler.py
            try:
                tof_stream.set("ToF::StreamFps", 2)
                tof_stream.set("ToF::Exposure", 0.1)
                tof_stream.set("ToF::Distance", 7.5)
                tof_stream.set("ToF::DepthMedianBlur", 0)
                tof_stream.set("ToF::DepthFlyingPixelRemoval", 1)
                tof_stream.set("ToF::Threshold", 100)
                tof_stream.set("ToF::Gain", 9)
                tof_stream.set("ToF::AutoExposure", 0)
                tof_stream.set("ToF::DepthSmoothStrength", 0)
                tof_stream.set("ToF::DepthCompletion", 0)
            except Exception:
                pass

            # Fetch cam params from selected stream; apply to transformer
            try:
                if tof_stream.getCamPara(intrinsic, extrinsic):
                    try:
                        vidu.print_intrinsics(intrinsic)
                    except Exception:
                        pass
                    fx = fy = cx = cy = None
                    k1 = k2 = p1 = p2 = k3 = 0.0
                    try:
                        attrs = {name.lower(): getattr(intrinsic, name) for name in dir(intrinsic)
                                 if not name.startswith('__') and not callable(getattr(intrinsic, name))}
                        fx = attrs.get('fx', attrs.get('f_x', attrs.get('f', None)))
                        fy = attrs.get('fy', attrs.get('f_y', None))
                        cx = attrs.get('cx', attrs.get('c_x', None))
                        cy = attrs.get('cy', attrs.get('c_y', None))
                        k1 = attrs.get('k1', attrs.get('k_1', 0.0))
                        k2 = attrs.get('k2', attrs.get('k_2', 0.0))
                        p1 = attrs.get('p1', 0.0)
                        p2 = attrs.get('p2', 0.0)
                        k3 = attrs.get('k3', attrs.get('k_3', 0.0))
                        if (fx is None or fy is None or cx is None or cy is None) and hasattr(intrinsic, 'K'):
                            K = getattr(intrinsic, 'K')
                            try:
                                K = np.array(K, dtype=float).reshape(3, 3)
                                fx, fy = K[0, 0], K[1, 1]
                                cx, cy = K[0, 2], K[1, 2]
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if all(v is not None for v in [fx, fy, cx, cy]):
                        try:
                            fxv, fyv, cxv, cyv = float(fx), float(fy), float(cx), float(cy)
                            Ktxt0 = f"[{fxv:.3f}, 0.000, {cxv:.3f}]"
                            Ktxt1 = f"[0.000, {fyv:.3f}, {cyv:.3f}]"
                            Ktxt2 = "[0.000, 0.000, 1.000]"
                            annotator.k_matrix_text = (Ktxt0, Ktxt1, Ktxt2)
                            # Apply to transformer
                            annotator.transformer.K = np.array([[fxv, 0.0, cxv], [0.0, fyv, cyv], [0.0, 0.0, 1.0]], dtype=np.float64)
                            annotator.transformer.dist = np.array([float(k1), float(k2), float(p1), float(p2), float(k3)], dtype=np.float64)
                        except Exception:
                            pass
            except Exception:
                pass

            print("Press 'q' to quit")
            while True:
                # ToF frame (used for both display and depth)
                depth_raw = None
                try:
                    images_tof = tof_stream.getPyImage()
                    # camera_handler expects 2 images and uses the second for ToF IR/depth
                    if images_tof and len(images_tof) >= 2 and images_tof[1] is not None and images_tof[1].size > 0:
                        depth_raw = images_tof[1]
                except Exception:
                    depth_raw = None

                if depth_raw is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Align orientation
                depth = cv2.flip(depth_raw, -1)
                annotator.latest_depth = depth

                # Track source size (before resize)
                annotator.set_source_size_from_frame(depth)

                # Build a visualization for ToF depth
                # Display visualization similar to camera_handler processing
                vis = depth
                try:
                    ir8 = cv2.convertScaleAbs(depth, alpha=(2000.0 / 60000.0))
                    _, ir8 = cv2.threshold(ir8, 10, 60, cv2.THRESH_TOZERO)
                    vis = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
                except Exception:
                    pass

                # Resize for display and annotate
                vis = annotator._process_rgb_image(vis)
                vis = annotator.overlay_annotation(vis)
                cv2.imshow(annotator.window_name, vis)
                time.sleep(2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        cv2.destroyAllWindows()
        return

    # RGB mode
    with vidu.PDstream(device, rgb_index) as rgb_stream:
        if not rgb_stream.init():
            print("RGB stream init failed")
            sys.exit(1)
        tof_stream = None
        if tof_index is not None:
            tof_stream = vidu.PDstream(device, tof_index)
            if not tof_stream.init():
                print("ToF stream init failed; continuing without ToF depth")
                tof_stream = None
            else:
                # Configure ToF stream parameters as in camera_handler
                try:
                    tof_stream.set("ToF::StreamFps", 100)
                    tof_stream.set("ToF::Exposure", 0.1)
                    tof_stream.set("ToF::Distance", 7.5)
                    tof_stream.set("ToF::DepthMedianBlur", 0)
                    tof_stream.set("ToF::DepthFlyingPixelRemoval", 1)
                    tof_stream.set("ToF::Threshold", 100)
                    tof_stream.set("ToF::Gain", 9)
                    tof_stream.set("ToF::AutoExposure", 0)
                    tof_stream.set("ToF::DepthSmoothStrength", 0)
                    tof_stream.set("ToF::DepthCompletion", 0)
                except Exception:
                    pass
        try:
            if rgb_stream.getCamPara(intrinsic, extrinsic):
                try:
                    vidu.print_intrinsics(intrinsic)
                except Exception:
                    pass
                fx = fy = cx = cy = None
                k1 = k2 = p1 = p2 = k3 = 0.0
                try:
                    attrs = {name.lower(): getattr(intrinsic, name) for name in dir(intrinsic)
                             if not name.startswith('__') and not callable(getattr(intrinsic, name))}
                    fx = attrs.get('fx', attrs.get('f_x', attrs.get('f', None)))
                    fy = attrs.get('fy', attrs.get('f_y', None))
                    cx = attrs.get('cx', attrs.get('c_x', None))
                    cy = attrs.get('cy', attrs.get('c_y', None))
                    k1 = attrs.get('k1', attrs.get('k_1', 0.0))
                    k2 = attrs.get('k2', attrs.get('k_2', 0.0))
                    p1 = attrs.get('p1', 0.0)
                    p2 = attrs.get('p2', 0.0)
                    k3 = attrs.get('k3', attrs.get('k_3', 0.0))
                    if (fx is None or fy is None or cx is None or cy is None) and hasattr(intrinsic, 'K'):
                        K = getattr(intrinsic, 'K')
                        try:
                            K = np.array(K, dtype=float).reshape(3, 3)
                            fx, fy = K[0, 0], K[1, 1]
                            cx, cy = K[0, 2], K[1, 2]
                        except Exception:
                            pass
                except Exception:
                    pass
                if all(v is not None for v in [fx, fy, cx, cy]):
                    try:
                        fxv, fyv, cxv, cyv = float(fx), float(fy), float(cx), float(cy)
                        Ktxt0 = f"[{fxv:.3f}, 0.000, {cxv:.3f}]"
                        Ktxt1 = f"[0.000, {fyv:.3f}, {cyv:.3f}]"
                        Ktxt2 = "[0.000, 0.000, 1.000]"
                        annotator.k_matrix_text = (Ktxt0, Ktxt1, Ktxt2)
                        # Apply to transformer
                        annotator.transformer.K = np.array([[fxv, 0.0, cxv], [0.0, fyv, cyv], [0.0, 0.0, 1.0]], dtype=np.float64)
                        annotator.transformer.dist = np.array([float(k1), float(k2), float(p1), float(p2), float(k3)], dtype=np.float64)
                    except Exception:
                        pass
        except Exception:
            pass

        print("Press 'q' to quit")
        while True:
            # RGB frame
            images_rgb = rgb_stream.getPyImage()
            if not images_rgb:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            rgb = None
            # Some RGB streams might also include depth; we take first as RGB
            depth_from_rgb = None
            for img in images_rgb:
                if img is not None and img.size > 0:
                    if rgb is None:
                        rgb = img
                    elif depth_from_rgb is None:
                        depth_from_rgb = img
                        break

            if rgb is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ToF depth (preferred if available)
            depth = None
            if tof_stream is not None:
                try:
                    images_tof = tof_stream.getPyImage()
                    if images_tof and len(images_tof) >= 2 and images_tof[1] is not None and images_tof[1].size > 0:
                        depth = images_tof[1]
                except Exception:
                    depth = None
            if depth is None:
                depth = depth_from_rgb

            # Align orientation
            rgb = cv2.flip(rgb, -1)
            annotator.latest_depth = cv2.flip(depth, -1) if depth is not None else None

            # Track source size (before resize)
            annotator.set_source_size_from_frame(rgb)

            # Resize RGB for display
            vis = annotator._process_rgb_image(rgb)

            vis = annotator.overlay_annotation(vis)
            cv2.imshow(annotator.window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if tof_stream is not None:
            try:
                tof_stream.close()
            except Exception:
                pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR/ToF click annotator")
    parser.add_argument("--tof", action="store_true", help="Display ToF depth as main view")
    args = parser.parse_args()
    main(use_tof=args.tof)


