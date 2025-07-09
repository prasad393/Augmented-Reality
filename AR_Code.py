#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# --- Constants ---
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_1000
WINDOW_NAME = 'Augmented Reality Output'
DEFAULT_MARKER_IMAGE_PATH = 'pictures/20221115_113424.jpg'
DEFAULT_OVERLAY_IMAGE_PATH = 'pictures/poster.jpg'
DEFAULT_OUTPUT_IMAGE_PATH = 'pictures/ar_output.png'
MARKER_SCALE_FACTOR = 1.0 # Default scale, can be adjusted

# --- Utility Functions ---

def load_image(image_path: str):
    """
    Loads an image from the specified path.
    Exits if the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}")
        exit()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        exit()
    return img

def display_image(image, window_name=WINDOW_NAME, keep_ratio=True):
    """Displays an image in a resizable window."""
    if keep_ratio:
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, output_path: str):
    """Saves the image to the specified path."""
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, image)
        print(f"Successfully saved image to {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

# --- Core AR Functions ---

def find_aruco_markers(image, aruco_dict_type=ARUCO_DICT_TYPE):
    """
    Detects ArUco markers in the given image.
    Returns marker corners and ids.
    """
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    return corners, ids, rejected_img_points

def scale_marker_corners(corners, scale_factor):
    """
    Scales the marker corners from their center.
    """
    if not corners or len(corners) == 0:
        return [] # Return empty if no corners

    # Assuming the first detected marker's corners are used
    marker_corners = np.array(corners[0][0], dtype=np.float32)

    if scale_factor == 1.0:
        return marker_corners # No scaling needed

    center = np.mean(marker_corners, axis=0)
    translated_corners = marker_corners - center
    scaled_corners = translated_corners * scale_factor
    final_corners = scaled_corners + center
    return final_corners

def plot_corners(original_corners, scaled_corners):
    """
    Plots the original and scaled marker corners using Matplotlib.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    if original_corners is not None and len(original_corners) > 0:
        ax[0].plot(original_corners[:, 0], original_corners[:, 1], '-o')
        ax[0].set_title('Original Marker Corners')
        ax[0].invert_yaxis()
        ax[0].set_aspect('equal', adjustable='box')


    if scaled_corners is not None and len(scaled_corners) > 0:
        ax[1].plot(scaled_corners[:, 0], scaled_corners[:, 1], '-o')
        ax[1].set_title(f'Scaled Marker Corners (Factor: {MARKER_SCALE_FACTOR})')
        ax[1].invert_yaxis()
        ax[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def augment_image(base_image, overlay_image, detected_marker_corners, scale_factor=1.0, plot_debug_corners=False):
    """
    Overlays the overlay_image onto the base_image at the location of the
    detected_marker_corners, scaled by scale_factor.
    """
    if not detected_marker_corners:
        print("No ArUco markers detected. Cannot perform augmentation.")
        return base_image

    # Use the corners of the first detected marker
    # These are the points in the base_image where the overlay will be warped to
    destination_corners_unscaled = np.array(detected_marker_corners[0][0], dtype=np.float32)

    if scale_factor != 1.0:
        scaled_destination_corners = scale_marker_corners(detected_marker_corners, scale_factor)
    else:
        scaled_destination_corners = destination_corners_unscaled

    if plot_debug_corners:
        plot_corners(destination_corners_unscaled, scaled_destination_corners if scale_factor != 1.0 else None)


    # Define corners of the overlay image (source points)
    overlay_h, overlay_w = overlay_image.shape[:2]
    source_corners = np.array([
        [0, 0],
        [overlay_w - 1, 0],
        [overlay_w - 1, overlay_h - 1],
        [0, overlay_h - 1]
    ], dtype=np.float32)

    # Calculate the perspective transformation matrix
    homography_matrix, status = cv2.findHomography(source_corners, scaled_destination_corners)
    if homography_matrix is None:
        print("Error: Could not compute homography matrix. Check marker detection and corner correspondence.")
        return base_image

    # Warp the overlay image to fit the marker's perspective
    warped_overlay = cv2.warpPerspective(overlay_image, homography_matrix,
                                         (base_image.shape[1], base_image.shape[0]))

    # Create a mask for the warped overlay
    mask = np.zeros_like(base_image, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(scaled_destination_corners)], (255, 255, 255))

    # Invert the mask to create a "hole" in the base image
    inverse_mask = cv2.bitwise_not(mask)
    base_image_holed = cv2.bitwise_and(base_image, inverse_mask)

    # Combine the holed base image with the warped overlay
    augmented_result = cv2.bitwise_or(base_image_holed, warped_overlay)

    return augmented_result

# --- Main Execution ---

def main():
    """
    Main function to run the AR application.
    Parses arguments, loads images, detects markers, and performs augmentation.
    """
    parser = argparse.ArgumentParser(description="Augmented Reality with ArUco Markers.")
    parser.add_argument("--marker_img", type=str, default=DEFAULT_MARKER_IMAGE_PATH,
                        help="Path to the image containing the ArUco marker.")
    parser.add_argument("--overlay_img", type=str, default=DEFAULT_OVERLAY_IMAGE_PATH,
                        help="Path to the image to overlay on the marker.")
    parser.add_argument("--output_img", type=str, default=DEFAULT_OUTPUT_IMAGE_PATH,
                        help="Path to save the augmented output image.")
    parser.add_argument("--scale", type=float, default=MARKER_SCALE_FACTOR,
                        help="Scaling factor for the overlay image relative to the marker size.")
    parser.add_argument("--show_output", action='store_true',
                        help="Display the augmented image in a window.")
    parser.add_argument("--plot_corners", action='store_true',
                        help="Plot original and scaled marker corners for debugging.")


    args = parser.parse_args()

    # Update global scale factor if provided via CLI
    global MARKER_SCALE_FACTOR
    MARKER_SCALE_FACTOR = args.scale

    # Load images
    marker_image = load_image(args.marker_img)
    overlay_image = load_image(args.overlay_img)

    # Detect ArUco markers in the marker_image
    detected_corners, ids, _ = find_aruco_markers(marker_image)

    if ids is None or len(ids) == 0:
        print(f"No ArUco markers found in {args.marker_img}. Exiting.")
        if args.show_output:
            display_image(marker_image, window_name="Original Image (No Markers Found)")
        return

    print(f"Detected {len(ids)} marker(s). Using the first detected marker for augmentation.")

    # Augment the image
    # The base_image for augmentation is a copy to avoid modifying the original marker_image
    final_image = augment_image(marker_image.copy(), overlay_image, detected_corners,
                                scale_factor=args.scale, plot_debug_corners=args.plot_corners)

    # Display or save the final image
    if args.show_output:
        display_image(final_image)

    save_image(final_image, args.output_img)

if __name__ == '__main__':
    main()
