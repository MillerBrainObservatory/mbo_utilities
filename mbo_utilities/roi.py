from enum import Enum

# SelectedROI ?
# ROI?
# CurrentROI?
class ROIMode(Enum):
    ALL = 0
    SINGLE = 1
    MULTIPLE = 2

@property
def roi_mode(self):
    # TODO: make this self.roi once Scan_MBO.roi is changed
    if self.selected_roi is None:
        return ROIMode.ALL
    elif isinstance(self.selected_roi, int):
        return ROIMode.SINGLE
    else:
        return ROIMode.MULTIPLE