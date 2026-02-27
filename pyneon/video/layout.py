import numpy as np
import pandas as pd

class Layout:
    def __init__(
        self,
        marker_centers: pd.DataFrame = None,
        surface_width: float = None,
        surface_height: float = None,
        surface_corners: np.ndarray = None,
        layout_dict: dict = None,
    ):
        self.marker_lookup = {}
        self.source = None
        
        # Initialize using the appropriate factory method
        if marker_centers is not None:
            self.from_marker_centers(marker_centers)
        elif surface_corners is not None:
            self.from_surface_corners(surface_corners)
        elif surface_width is not None and surface_height is not None:
            self.from_surface_size(surface_width, surface_height)
        elif layout_dict is not None:
            self.from_layout_dict(layout_dict)

    def from_marker_centers(self, layout: pd.DataFrame):
        """
        This method expects a marker layout dataframe with columns: "marker id", "center x", "center y", and "size". It computes the corners of each marker based on the center and size, and stores them in a dictionary keyed by marker id.
        """
        #validate that layout has the required columns
        required_columns = ["marker id", "center x", "center y", "size"]
        for col in required_columns:
            if col not in layout.columns:
                raise ValueError(f"Layout must contain '{col}' column.")
            
        marker_layout = _prepare_marker_layout(layout)
        self.marker_lookup = _find_reference_lookup(marker_layout, id_col="marker id")
        self.source = "marker"

    def from_surface_size(self, surface_width: float, surface_height: float):
        """
        This method expects a surface layout dataframe with columns: "surface id", "top left x", "top left y", "top right x", "top right y", "bottom right x", "bottom right y", "bottom left x", and "bottom left y". It stores the corners of each surface in a dictionary keyed by surface id.
        """
        surface_layout = pd.DataFrame({
            "surface id": [0],
            "corners": [np.array([
                [0, 0],
                [surface_width, 0],
                [surface_width, surface_height],
                [0, surface_height],
            ], dtype=np.float32)],
        })
        self.marker_lookup = _find_reference_lookup(surface_layout, id_col="surface id")
        self.source = "surface"

    def from_surface_corners(self, corners: np.ndarray):
        """
        This method expects a surface layout dataframe with columns: "surface id", "top left x", "top left y", "top right x", "top right y", "bottom right x", "bottom right y", "bottom left x", and "bottom left y". It stores the corners of each surface in a dictionary keyed by surface id.
        """
        if corners.shape != (4, 2):
            raise ValueError("Corners must be a 4x2 array representing the four corners of the surface.")

        surface_layout = pd.DataFrame({
            "surface id": [0],
            "corners": [corners],
        })
        self.marker_lookup = _find_reference_lookup(surface_layout, id_col="surface id")
        self.source = "surface"

    def from_layout_dict(self, layout_dict: dict):
        self.marker_lookup = layout_dict

def _prepare_marker_layout(marker_layout: pd.DataFrame) -> pd.DataFrame:
    marker_layout = marker_layout.copy()
    marker_layout["corners"] = marker_layout.apply(
        lambda row: np.array(
            [
            [
                row["center x"] - row["size"] / 2,
                row["center y"] - row["size"] / 2,
            ],
            [
                row["center x"] + row["size"] / 2,
                row["center y"] - row["size"] / 2,
            ],
            [
                row["center x"] + row["size"] / 2,
                row["center y"] + row["size"] / 2,
            ],
            [
                row["center x"] - row["size"] / 2,
                row["center y"] + row["size"] / 2,
            ],
        ],
        dtype=np.float32,
    ),
    axis=1,
    )
    return marker_layout

def _find_reference_lookup(layout: pd.DataFrame, id_col: str) -> dict:
    
    if id_col not in layout.columns:
        raise ValueError(f"Layout must contain '{id_col}' column for reference lookup.")
    
    lookup = {}
    for _, row in layout.iterrows():
        marker_id = row[id_col]
        if marker_id in lookup:
            raise ValueError(f"Duplicate {id_col} {marker_id} found in layout.")
        lookup[marker_id] = row["corners"]
    
    return lookup