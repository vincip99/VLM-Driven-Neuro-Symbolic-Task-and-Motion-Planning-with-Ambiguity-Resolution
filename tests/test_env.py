import robosuite as suite
import cv2
import numpy as np

from src.envs import TaskSorting

def test_environment():
    # initialize the environment
    env = suite.make(
        env_name="TaskSorting",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["top_down_vlm", "frontal_debug"],
        camera_heights=512,
        camera_widths=512,
    )

    obs = env.reset()

    # Extract and save the images
    for cam_name in ["top_down_vlm", "frontal_debug"]:
        try:
            # Robosuite returns numpy arrays in RGB format. 
            rgb_image = obs[f"{cam_name}_image"]
            
            # If your saved images are flipped, uncomment the next line:
            rgb_image = np.flipud(rgb_image)
            
            # OpenCV expects BGR format for saving files
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            filename = f"render_{cam_name}.jpg"
            cv2.imwrite(filename, bgr_image)
            print(f"Successfully saved: {filename}")
            
        except KeyError:
            print(f"Error: Could not find camera '{cam_name}'. Check your arena XML strings.")

    print("\nTest complete! Open the saved JPEGs to verify the framing.")
    env.close()

if __name__ == "__main__":
    test_environment()