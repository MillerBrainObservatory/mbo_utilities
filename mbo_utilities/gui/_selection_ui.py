"""
shared ui components for timepoint and z-plane selection.

used by both save-as dialog and suite2p run tab.
"""

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.arrays.features._slicing import parse_timepoint_selection


def _parse_z_selection(text: str, num_planes: int) -> tuple[int, int, int, str]:
    """
    Parse z-plane selection string like "1:14" or "1:14:2".

    Returns (start, stop, step, error_msg). error_msg is empty on success.
    """
    text = text.strip()
    if not text:
        return 1, num_planes, 1, ""

    parts = text.split(":")
    try:
        if len(parts) == 1:
            # single value
            val = int(parts[0])
            if val < 1 or val > num_planes:
                return 1, num_planes, 1, f"plane {val} out of range 1-{num_planes}"
            return val, val, 1, ""
        elif len(parts) == 2:
            start = int(parts[0])
            stop = int(parts[1])
            if start < 1:
                start = 1
            if stop > num_planes:
                stop = num_planes
            if start > stop:
                return 1, num_planes, 1, "start > stop"
            return start, stop, 1, ""
        elif len(parts) == 3:
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2])
            if start < 1:
                start = 1
            if stop > num_planes:
                stop = num_planes
            if step < 1:
                step = 1
            if start > stop:
                return 1, num_planes, 1, "start > stop"
            return start, stop, step, ""
        else:
            return 1, num_planes, 1, "format: start:stop or start:stop:step"
    except ValueError:
        return 1, num_planes, 1, "invalid number"


def draw_selection_table(
    parent,
    max_frames: int,
    num_planes: int,
    tp_attr: str = "_saveas_tp",
    z_attr: str = "_saveas_z",
    id_suffix: str = "",
    suffix_attr: str | None = None,
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
    suffix_attr : str | None
        If provided, adds a Suffix row with this attribute name for output suffix.
    """
    # get/set attributes dynamically
    tp_selection = getattr(parent, f"{tp_attr}_selection", f"1:{max_frames}")
    tp_error = getattr(parent, f"{tp_attr}_error", "")
    tp_parsed = getattr(parent, f"{tp_attr}_parsed", None)

    # z-plane selection as text (convert from old start/stop/step if needed)
    z_selection_attr = f"{z_attr}_selection"
    z_error_attr = f"{z_attr}_error"
    if not hasattr(parent, z_selection_attr):
        # initialize from old format or default
        z_start = getattr(parent, f"{z_attr}_start", 1)
        z_stop = getattr(parent, f"{z_attr}_stop", num_planes)
        z_step = getattr(parent, f"{z_attr}_step", 1)
        if z_step == 1:
            z_selection = f"{z_start}:{z_stop}"
        else:
            z_selection = f"{z_start}:{z_stop}:{z_step}"
        setattr(parent, z_selection_attr, z_selection)
        setattr(parent, z_error_attr, "")

    z_selection = getattr(parent, z_selection_attr)
    z_error = getattr(parent, z_error_attr, "")

    # parse current z selection
    z_start, z_stop, z_step, _ = _parse_z_selection(z_selection, num_planes)

    INPUT_WIDTH = hello_imgui.em_size(13)

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
        imgui.set_next_item_width(INPUT_WIDTH)

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
            imgui.set_next_item_width(INPUT_WIDTH)

            # red border if error
            had_z_error = bool(z_error)
            if had_z_error:
                imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.3, 0.1, 0.1, 1.0))

            changed, new_val = imgui.input_text(f"##z{id_suffix}", z_selection)
            if changed:
                setattr(parent, z_selection_attr, new_val)
                z_start, z_stop, z_step, err = _parse_z_selection(new_val, num_planes)
                setattr(parent, z_error_attr, err)
                z_error = err
                # update old-style attrs for compatibility
                setattr(parent, f"{z_attr}_start", z_start)
                setattr(parent, f"{z_attr}_stop", z_stop)
                setattr(parent, f"{z_attr}_step", z_step)

            if had_z_error:
                imgui.pop_style_color()

            if z_error and imgui.is_item_hovered():
                imgui.set_tooltip(z_error)

            imgui.table_next_column()
            if imgui.small_button(f"All##z{id_suffix}"):
                setattr(parent, z_selection_attr, f"1:{num_planes}")
                setattr(parent, z_error_attr, "")
                setattr(parent, f"{z_attr}_start", 1)
                setattr(parent, f"{z_attr}_stop", num_planes)
                setattr(parent, f"{z_attr}_step", 1)
                z_start, z_stop, z_step = 1, num_planes, 1
                z_error = ""

            imgui.table_next_column()
            selected_planes = list(range(z_start, z_stop + 1, z_step))
            n_planes_selected = len(selected_planes)
            if z_error:
                imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "invalid")
            else:
                imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{n_planes_selected}/{num_planes}")

        # suffix row (optional)
        if suffix_attr:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Suffix")

            imgui.table_next_column()
            imgui.set_next_item_width(INPUT_WIDTH)
            suffix_val = getattr(parent, suffix_attr, "")
            changed, new_suffix = imgui.input_text(f"##suffix{id_suffix}", suffix_val)
            if changed:
                setattr(parent, suffix_attr, new_suffix)

            imgui.table_next_column()
            # empty - no "All" button for suffix

            imgui.table_next_column()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 25.0)
                imgui.text_unformatted(
                    "Optional suffix for output filenames.\n"
                    "Examples: 'processed', 'session1'"
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

        imgui.end_table()

    # return parsed selection info for caller
    return tp_parsed, z_start, z_stop, z_step
