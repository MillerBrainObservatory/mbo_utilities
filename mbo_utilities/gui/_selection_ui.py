"""
shared ui components for timepoint and z-plane selection.

used by both save-as dialog and suite2p run tab.
"""

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.arrays.features._slicing import parse_timepoint_selection


def draw_selection_table(
    parent,
    max_frames: int,
    num_planes: int,
    tp_attr: str = "_saveas_tp",
    z_attr: str = "_saveas_z",
    id_suffix: str = "",
):
    """
    Draw a selection table for timepoints and z-planes.

    Parameters
    ----------
    parent : Any
        Parent widget with selection state attributes.
    max_frames : int
        Maximum number of frames in data.
    num_planes : int
        Number of z-planes in data.
    tp_attr : str
        Attribute prefix for timepoint state (e.g., "_saveas_tp" or "_s2p_tp").
    z_attr : str
        Attribute prefix for z-plane state (e.g., "_saveas_z" or "_s2p_z").
    id_suffix : str
        Suffix for imgui IDs to avoid conflicts.
    """
    # get/set attributes dynamically
    tp_selection = getattr(parent, f"{tp_attr}_selection", f"1:{max_frames}")
    tp_error = getattr(parent, f"{tp_attr}_error", "")
    tp_parsed = getattr(parent, f"{tp_attr}_parsed", None)

    z_start = getattr(parent, f"{z_attr}_start", 1)
    z_stop = getattr(parent, f"{z_attr}_stop", num_planes)
    z_step = getattr(parent, f"{z_attr}_step", 1)

    table_flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    if imgui.begin_table(f"selection_table{id_suffix}", 4, table_flags):
        # column widths for alignment
        imgui.table_setup_column("dim", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(7))
        imgui.table_setup_column("input", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(14))
        imgui.table_setup_column("all", imgui.TableColumnFlags_.width_fixed, hello_imgui.em_size(3))
        imgui.table_setup_column("info", imgui.TableColumnFlags_.width_stretch)

        # timepoints row
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text("Timepoints")

        imgui.table_next_column()
        imgui.set_next_item_width(hello_imgui.em_size(13))

        # red border if error
        had_error = bool(tp_error)
        if had_error:
            imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.3, 0.1, 0.1, 1.0))

        changed, new_val = imgui.input_text(f"##tp{id_suffix}", tp_selection)
        if changed:
            setattr(parent, f"{tp_attr}_selection", new_val)
            try:
                parsed = parse_timepoint_selection(new_val, max_frames)
                setattr(parent, f"{tp_attr}_parsed", parsed)
                setattr(parent, f"{tp_attr}_error", "")
                tp_parsed = parsed
                tp_error = ""
            except ValueError as e:
                setattr(parent, f"{tp_attr}_error", str(e))
                setattr(parent, f"{tp_attr}_parsed", None)
                tp_error = str(e)
                tp_parsed = None

        if had_error:
            imgui.pop_style_color()

        if tp_error and imgui.is_item_hovered():
            imgui.set_tooltip(tp_error)

        imgui.table_next_column()
        if imgui.small_button(f"All##tp{id_suffix}"):
            setattr(parent, f"{tp_attr}_selection", f"1:{max_frames}")
            parsed = parse_timepoint_selection(f"1:{max_frames}", max_frames)
            setattr(parent, f"{tp_attr}_parsed", parsed)
            setattr(parent, f"{tp_attr}_error", "")
            tp_parsed = parsed

        imgui.table_next_column()
        # frame count info
        if tp_parsed:
            n_frames = tp_parsed.count
            if tp_parsed.exclude_str:
                n_excluded = len(tp_parsed.exclude_indices)
                imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{n_frames}/{max_frames}")
                imgui.same_line()
                imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.4, 1.0), f"(-{n_excluded})")
            else:
                imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{n_frames}/{max_frames}")
        elif tp_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "invalid")
        else:
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"?/{max_frames}")

        # z-planes row (only if multi-plane)
        if num_planes > 1:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Z-Planes")

            imgui.table_next_column()
            # three inputs in a row: start : stop : step
            input_w = hello_imgui.em_size(3.5)
            colon_spacing = 4

            imgui.set_next_item_width(input_w)
            changed, val = imgui.input_int(f"##z_start{id_suffix}", z_start, step=0)
            if changed:
                z_start = max(1, min(val, z_stop))
                setattr(parent, f"{z_attr}_start", z_start)

            imgui.same_line(0, colon_spacing)
            imgui.text(":")
            imgui.same_line(0, colon_spacing)

            imgui.set_next_item_width(input_w)
            changed, val = imgui.input_int(f"##z_stop{id_suffix}", z_stop, step=0)
            if changed:
                z_stop = max(z_start, min(val, num_planes))
                setattr(parent, f"{z_attr}_stop", z_stop)

            imgui.same_line(0, colon_spacing)
            imgui.text(":")
            imgui.same_line(0, colon_spacing)

            imgui.set_next_item_width(input_w)
            changed, val = imgui.input_int(f"##z_step{id_suffix}", z_step, step=0)
            if changed:
                z_step = max(1, min(val, z_stop - z_start + 1))
                setattr(parent, f"{z_attr}_step", z_step)

            imgui.table_next_column()
            if imgui.small_button(f"All##z{id_suffix}"):
                setattr(parent, f"{z_attr}_start", 1)
                setattr(parent, f"{z_attr}_stop", num_planes)
                setattr(parent, f"{z_attr}_step", 1)
                z_start, z_stop, z_step = 1, num_planes, 1

            imgui.table_next_column()
            selected_planes = list(range(z_start, z_stop + 1, z_step))
            n_planes = len(selected_planes)
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{n_planes}/{num_planes}")

        imgui.end_table()

    # return parsed selection info for caller
    return tp_parsed, z_start, z_stop, z_step
