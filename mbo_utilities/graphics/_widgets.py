import numpy as np
from imgui_bundle import imgui, implot


def imgui_dynamic_table(
    table_id: str, data_lists: list, titles: list = None, selected_index: int = None
):
    """
    imgui dynamic table helper

    Parameters
    ----------
    table_id : str
        Unique identifier for the table.
    data_lists : list of lists
        A list of columns, where each inner list represents a column's data.
        All columns must have the same length.
    titles : list of str, optional
        Column titles corresponding to `data_lists`. If None, default names are assigned as "Column N".
    selected_index : int, optional
        Index of the row to highlight. Default is None (no highlighting).

    Raises
    ------
    ValueError
        If `data_lists` is empty or columns have inconsistent lengths.
        If `titles` is provided but does not match the number of columns.
    """

    if not data_lists or any(len(col) != len(data_lists[0]) for col in data_lists):
        raise ValueError(
            "data_lists columns must have consistent lengths and cannot be empty."
        )

    num_columns = len(data_lists)
    if titles is None:
        titles = [f"Column {i + 1}" for i in range(num_columns)]
    elif len(titles) != num_columns:
        raise ValueError(
            "Number of titles must match the number of columns in data_lists."
        )

    if imgui.begin_table(
        table_id,
        num_columns,
        flags=imgui.TableFlags_.borders | imgui.TableFlags_.resizable,  # noqa
    ):  # noqa
        for title in titles:
            imgui.table_setup_column(title, imgui.TableColumnFlags_.width_stretch)  # noqa

        for title in titles:
            imgui.table_next_column()
            imgui.text(title)

        for i, row_values in enumerate(zip(*data_lists)):
            imgui.table_next_row()
            for value in row_values:
                imgui.table_next_column()
                if i == selected_index:
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0, 250, 35, 1))  # noqa

                imgui.set_cursor_pos_x(
                    imgui.get_cursor_pos_x()
                    + imgui.get_column_width()
                    - imgui.calc_text_size(f"{int(value)}").x
                    - imgui.get_style().item_spacing.x
                )
                imgui.text(f"{int(value)}")

                if i == selected_index:
                    imgui.pop_style_color()

        imgui.end_table()


def implot_pollen(pollen_offsets, offset_store, zstack):
    imgui.begin_child("Z-Stack Analysis")

    z_planes = np.arange(len(pollen_offsets)) + 1

    # Offset Comparison Plot
    if implot.begin_plot("Offset Comparison Across Z-Planes"):
        implot.plot_bars(
            "Calibration Offsets",
            z_planes,
            np.array([p[0] for p in pollen_offsets]),
            bar_size=0.4,
        )
        implot.plot_bars(  # noqa
            "Selected Offsets",
            z_planes,
            np.array([s[0] for s in offset_store]),
            bar_size=0.4,
            shift=0.2,
        )
        implot.end_plot()

    # Mean Intensity Plot
    if implot.begin_plot("Mean Intensity Across Z-Planes"):
        mean_intensities = np.array(
            [np.mean(zstack[z]) for z in range(zstack.shape[0])]
        )
        implot.plot_line("Mean Intensity", z_planes, mean_intensities)
        implot.end_plot()

    # Cumulative Offset Drift Plot
    if implot.begin_plot("Cumulative Offset Drift"):
        cumulative_shifts = np.cumsum(np.array([s[0] for s in offset_store]))
        implot.plot_line("Cumulative Shift", z_planes, cumulative_shifts)
        implot.end_plot()

    imgui.end_child()
