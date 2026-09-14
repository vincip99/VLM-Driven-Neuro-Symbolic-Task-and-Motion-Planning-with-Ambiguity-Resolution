from robosuite.models.arenas import TableArena
import xml.etree.ElementTree as ET

class CustomArena(TableArena):
    """
    Custom arena class to modify lighting and cameras
    """
    def __init__(
            self,
            table_full_size=(1.4, 0.8, 0.05),
            table_offset=(0, 0, 0.8),
            has_legs=False
        ):
        # Initialize inherit class TableArena from Robosuite/models/arenas
        super().__init__(
            table_full_size=table_full_size,
            table_offset=table_offset,
            has_legs=has_legs
        )

        # add camera and light properties
        self._set_shadowless_lighting()
        self._add_cameras()

    def _set_shadowless_lighting(self):
        """Removes harsh default lights and adds diffuse ambient lighting."""
        for light in self.worldbody.findall("./light"):
            self.worldbody.remove(light)
            
        ambient_light = ET.Element(
                "light",
                name="vlm_ambient",
                dir="0 0 -1",
                pos="0 0 3.0",
                castshadow="false", 
                diffuse="0.8 0.8 0.8",
                specular="0.1 0.1 0.1",
            )
        self.worldbody.append(ambient_light)

    def _add_cameras(self):
        """Adds both the VLM top-down camera and the human debug camera."""
        # Top-Down VLM Camera (Placed 1 meter above the table)
        cam_height = self.table_offset[2] + 1.0
        top_down_cam = ET.Element(
            "camera",
            name="top_down_vlm",
            pos=f"0 0 {cam_height}",
            mode="fixed",
            quat="1 0 0 0" 
        )
        self.worldbody.append(top_down_cam)

        # Frontal Debug Camera (Placed in front of the table, angled down)
        frontal_cam = ET.Element(
            "camera",
            name="frontal_debug",
            pos="1.2 0 1.2", 
            mode="fixed",
            quat="0.560842 0.430459 0.430459 0.560842" 
        )
        self.worldbody.append(frontal_cam)