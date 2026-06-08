import os
from datetime import datetime
import numpy as np
import vtk

def export_screenshot(visualizer, filename=None, transparent=True, force_white_bg=False, scale_factor=3):
    """
    Captures a flawless, hyper-clean high-resolution screenshot.
    Scans every single subplot (renderer) to ensure absolutely NO text, 
    titles, or widgets remain in the final publication image.
    """
    if not hasattr(visualizer, "_active_plotter") or not visualizer._active_plotter:
        print("[-] Error: No active plotter found.")
        return

    plotter = visualizer._active_plotter
    render_window = plotter.render_window
    
    if render_window is None:
        print("[-] Error: VTK Render Window is not initialized.")
        return

    if filename is None:
        os.makedirs("captured_events", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("captured_events", f"publication_pure_{timestamp}.png")

    print(f"[*] Extracting 100% clean frame across all subplots (Scale x{scale_factor})...")

    # 1. Save original background
    original_bg = plotter.background_color
    
    # 2. Total cleanup of each subplot (renderer)
    hidden_elements = []
    
    # Retrieve the full collection of renderers in the scene
    renderer_collection = render_window.GetRenderers()
    renderer_collection.InitTraversal()
    
    while True:
        renderer = renderer_collection.GetNextItem()
        if renderer is None:
            break
            
        # A. Hide ALL 2D actors (includes LEGO Plot text and the HUD)
        for actor in renderer.GetActors2D():
            if actor.GetVisibility():
                actor.VisibilityOff()
                hidden_elements.append(actor)
                
        

    # 3. Apply background filters for export
    if force_white_bg:
        plotter.set_background("white")
    elif transparent:
        plotter.set_background("white" if not visualizer.theme.get("dark", True) else "black")

    try:
        # 4. Off-screen render in memory
        render_window.SetOffScreenRendering(1)
        render_window.Render()

        # 5. High-resolution capture pipeline
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(render_window)
        w2if.SetScale(scale_factor)
        
        if transparent and not force_white_bg:
            w2if.SetInputBufferTypeToRGBA()
        else:
            w2if.SetInputBufferTypeToRGB()
            
        w2if.ReadFrontBufferOff()
        w2if.Update()

        # Write the PNG file
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

        dims = w2if.GetOutput().GetDimensions()
        print(f"[+] Perfect raw image saved: {filename} ({dims[0]}x{dims[1]})")
        
    except Exception as e:
        print(f"[-] Failed to export high-res screenshot: {e}")
        
    finally:
        # 6. Complete restoration of all subplots
        for element in hidden_elements:
            element.VisibilityOn()


        # Properly release off-screen mode
        render_window.SetOffScreenRendering(0)
        plotter.background_color = original_bg
        
        print("[*] Interactive window fully restored (All subplots text re-activated).")

def export_interactive_video(visualizer, filename=None, *args, **kwargs):
    """
    CINEMATIC CONTINUOUS INTERPOLATION EXPORT.
    Fixes track stuttering by mathematically interpolating particle positions
    between discrete simulation steps, ensuring 100% fluid vector growth.
    """
    import os
    from datetime import datetime
    import numpy as np
    import pyvista as pv
    import imageio

    if filename is None:
        os.makedirs("captured_events", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("captured_events", f"cinematic_perfect_tracks_{timestamp}.mp4")

    print(f"\n[*] INITIATING CONTINUOUS TRACK INTERPOLATION EXPORT...")

    plotter = visualizer._active_plotter
    mode = visualizer._current_mode if hasattr(visualizer, "_current_mode") else "both"

    fps = 60
    duration_seconds = 5.0
    total_frames = int(duration_seconds * fps)

    start_r = -3.0
    max_detector_r = visualizer.detector_muon_r
    end_r = max(max_detector_r + 2.0, visualizer.calorimeter_outer_radius + 2.5)
    total_range = end_r - start_r

    ctx = visualizer._current_ctx if hasattr(visualizer, "_current_ctx") else {}

    video_writer = imageio.get_writer(
        filename, fps=fps, format='FFMPEG', mode='I',
        macro_block_size=None, quality=9, pixelformat='yuv420p'
    )

    try:
        for frame_idx in range(total_frames):
            # Smooth temporal progression (Smooth Step)
            t = frame_idx / (total_frames - 1)
            smooth_t = t * t * (3.0 - 2.0 * t)
            current_r = start_r + (total_range * smooth_t)

            # --- UPDATE DETECTOR GEOMETRY ZONE ---
            if mode in ["both", "detector"]:
                plotter.subplot(0, 0)
                if current_r < 0:
                    ctx["beam1_actor"].SetVisibility(True)
                    ctx["beam2_actor"].SetVisibility(True)
                    ctx["vertex_actor"].SetVisibility(True)
                    ctx["shockwave_actor"].SetVisibility(False)
                    if ctx["met_actor_line"]: ctx["met_actor_line"].SetVisibility(False)
                    if ctx["met_actor_tip"]: ctx["met_actor_tip"].SetVisibility(False)
                    for act in ctx["particle_actors"]: act.SetVisibility(False)
                    for act in ctx["jet_actors"]: act.SetVisibility(False)
                    for idx, act in visualizer._actor_registry["jet_towers"].items(): act["actor"].SetVisibility(False)

                    dist = abs(current_r)
                    beam_len = min(1.0, dist * 0.5)
                    ctx["beam1_actor"].mapper.dataset.points = np.array([[0.0, 0.0, dist + beam_len], [0.0, 0.0, dist]])
                    ctx["beam2_actor"].mapper.dataset.points = np.array([[0.0, 0.0, -(dist + beam_len)], [0.0, 0.0, -dist]])
                    ctx["beam1_actor"].mapper.dataset.Modified()
                    ctx["beam2_actor"].mapper.dataset.Modified()
                else:
                    ctx["beam1_actor"].SetVisibility(False)
                    ctx["beam2_actor"].SetVisibility(False)
                    if current_r >= visualizer.detector_ecal_r:
                        ctx["calorimeter_actor"].GetProperty().SetOpacity(0.06)

                    # --- CRITICAL TRACK INTERPOLATION ---
                    for i, (poly, full_path) in enumerate(ctx["particle_polydata_lists"]):
                        actor = ctx["particle_actors"][i]
                        distances = np.linalg.norm(full_path, axis=1)

                        # Points strictly inside the current wavefront
                        inside_mask = distances <= current_r
                        visible_points = list(full_path[inside_mask])

                        # Find the index of the first point outside the radius
                        outside_indices = np.where(~inside_mask)[0]

                        if len(outside_indices) > 0 and len(visible_points) > 0:
                            next_idx = outside_indices[0]
                            prev_idx = next_idx - 1

                            # Control points for linear interpolation
                            p_prev = full_path[prev_idx]
                            p_next = full_path[next_idx]
                            d_prev = distances[prev_idx]
                            d_next = distances[next_idx]

                            # Compute exact fraction (interpolation between simulation steps)
                            if d_next != d_prev:
                                frac = (current_r - d_prev) / (d_next - d_prev)
                                # Exact virtual point located on the sphere of radius current_r
                                exact_front_point = p_prev + frac * (p_next - p_prev)
                                visible_points.append(exact_front_point)

                        # Render the smoothed spline over the completed point set
                        if len(visible_points) > 1:
                            actor.SetVisibility(True)
                            # Increase spline point density to smooth angles
                            new_spline = pv.Spline(np.array(visible_points), n_points=max(20, len(visible_points) * 3))
                            actor.mapper.dataset.copy_from(new_spline)
                            actor.mapper.dataset.Modified()

                    if current_r >= visualizer.tracker_radius:
                        for act in ctx["jet_actors"]: act.SetVisibility(True)

                    if current_r >= (visualizer.calorimeter_outer_radius + 1.0):
                        if ctx["met_actor_line"]: ctx["met_actor_line"].SetVisibility(True)
                        if ctx["met_actor_tip"]: ctx["met_actor_tip"].SetVisibility(True)

            # --- LEGO PLOT UPDATE ---
            if mode in ["both", "lego"]:
                v_current_r = current_r if mode == "both" else (current_r + 3.0)
                plotter.subplot(0, 0) if mode == "lego" else plotter.subplot(0, 1)

                if v_current_r >= visualizer.detector_ecal_r:
                    progress = (v_current_r - visualizer.detector_ecal_r) / (visualizer.calorimeter_outer_radius - visualizer.detector_ecal_r)
                    for i, (box_mesh, init_pts) in enumerate(ctx["lego_mesh_references"]):
                        target_height = ctx["max_heights"][i] * min(1.0, max(0.0, progress))
                        if target_height > 0.001:
                            visualizer._actor_registry["jet_towers"][i]["actor"].SetVisibility(True)
                            new_pts = init_pts.copy()
                            new_pts[init_pts[:, 2] > 0.001, 2] = target_height
                            box_mesh.points = new_pts

            # Rendu et capture
            plotter.render()
            screenshot_matrix = plotter.screenshot(None, return_img=True)
            video_writer.append_data(screenshot_matrix)

    except Exception as e:
        print(f"[-] Video production failed: {e}")
    finally:
        video_writer.close()
        plotter.render()
        print("[*] Unified fluid export complete.")

def export_html(visualizer, filename=None, force_clean_scene=True, force_white_bg=False):
    """
    Exports the complete active 3D scene into a standalone interactive HTML file.
    Follows the strict visualizer pipeline architecture to sweep text or widgets 
    before freezing the WebGL geometry, then fully restores the canvas state.
    """
    if not hasattr(visualizer, "_active_plotter") or not visualizer._active_plotter:
        print("[-] Error: No active plotter found.")
        return

    plotter = visualizer._active_plotter
    render_window = plotter.render_window
    
    if render_window is None:
        print("[-] Error: VTK Render Window is not initialized.")
        return

    if filename is None:
        os.makedirs("exported_scenes", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("exported_scenes", f"interactive_event_{timestamp}.html")

    print(f"[*] Packaging 100% responsive WebGL asset across all views...")

    # 1. Save original graphical environment
    original_bg = plotter.background_color
    hidden_elements = []

    # 2. Clinical cleanup of each subplot (renderer) if required
    # Avoid font/widget artifacts frozen into the HTML mesh
    if force_clean_scene:
        renderer_collection = render_window.GetRenderers()
        renderer_collection.InitTraversal()
        
        while True:
            renderer = renderer_collection.GetNextItem()
            if renderer is None:
                break
                
            # Hide text labels, metadata banners and HUD
            for actor in renderer.GetActors2D():
                if actor.GetVisibility():
                    actor.VisibilityOff()
                    hidden_elements.append(actor)

    # 3. Adjust background for web exposure
    if force_white_bg:
        plotter.set_background("white")

    try:
        # In VTK/PyVista, exporting composite scenes uses the native scene exporter
        # which serializes the global geometric scene into a standard HTML package
        print(f"[*] Compiling WebGL geometries to standalone HTML document...")
        
        # 4. Generate and write the standalone HTML file via PyVista exporter
        plotter.export_html(filename)
        
        print(f"[+] Interactive 3D event deployment ready: {filename}")
        print(f"    -> Can be shared instantly. Open with any browser (Chrome/Safari/Firefox).")

    except Exception as e:
        print(f"[-] Failed to export interactive HTML framework: {e}")
        
    finally:
        # 5. Full restoration of the interactive sequence
        if force_clean_scene:
            for element in hidden_elements:
                element.VisibilityOn()

        # Restore visualizer background
        plotter.background_color = original_bg
        plotter.render()
        
        print("[*] Interactive window fully restored (HUD re-anchored to the live viewport).")