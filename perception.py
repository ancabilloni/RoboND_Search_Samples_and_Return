import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160), color=1):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = color
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def pix_to_world(xpix, ypix, x_rover, y_rover, yaw_rover, world_size, scale):
    # Map pixels from rover space to world coords
    yaw = yaw_rover * np.pi / 180
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_((((xpix * np.cos(yaw)) - (ypix * np.sin(yaw)))/scale) + x_rover), 
                            0, world_size - 1)
    y_pix_world = np.clip(np.int_((((xpix * np.sin(yaw)) + (ypix * np.cos(yaw)))/scale) + y_rover), 
                            0, world_size - 1)
  
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

# Define a function to find obstacle on warp persepctive
def thresh_obstacle(img, rgb_thresh=(160,160,160), color = 255):
    color_select = np.zeros_like(img[:,:,0])
    threshold = (img[:,:,0] < rgb_thresh[0])\
                & (img[:,:,1] < rgb_thresh[1])\
                & (img[:,:,2] < rgb_thresh[2])
    color_select[threshold] = color
    return color_select

# Define a function to find item of interest on warp perspective
def interested_object(img, rgb_thresh=(255,255,150), color = 255):
    color_select = np.zeros_like(img[:,:,0])
    threshold = (img[:,:,0] < rgb_thresh[0])\
                & (img[:,:,1] < rgb_thresh[1]) & (img[:,:,1] >120)\
                & (img[:,:,2] < rgb_thresh[2])
    color_select[threshold] = color
    return color_select


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    image = Rover.img
    dst_size = 5
    bottom_offset = 6
    src = np.float32([[14,140], [301, 140], [200, 96], [118, 96]])
    dst = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size],
                    [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size]])
    # 2) Apply perspective transform
    warped_img = perspect_transform(image, src, dst)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    apply_thres = color_thresh(img = warped_img, color = 1)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    Rover.vision_image[:,:,0] = thresh_obstacle(warped_img)
    Rover.vision_image[:,:,1] = color_thresh(img = warped_img, color = 255)
    Rover.vision_image[:,:,2] = interested_object(warped_img)
    # 5) Convert map image pixel values to rover-centric coords
    x_pix, y_pix = rover_coords(apply_thres)
    # 6) Convert rover-centric pixel values to world coordinates
    x_world, y_world = pix_to_world(x_pix, y_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], scale = 10)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 0] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 0] += 1
    # World Navigable
    Rover.worldmap[y_world, x_world, 2] = 255 #navigable
    # World Obstacle
    obstacle_x_pix, obstacle_y_pix = rover_coords(thresh_obstacle(img = warped_img, color = 1))
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_x_pix, obstacle_y_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], scale = 10)
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255

    # World Rock
    rock_x_pix, rock_y_pix = rover_coords(interested_object(img = warped_img, color = 1))
    rock_x_world, rock_y_world = pix_to_world(rock_x_pix, rock_y_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], scale = 10)
    Rover.worldmap[rock_y_world, rock_x_world, 1] = 255

    # 8) Convert rover-centric pixel positions to polar coordinates
    distance, angles = to_polar_coords(x_pix, y_pix)
    # mean_dir = np.mean(angles)
    # mean_dist = np.mean(distance)
    # Update Rover pixel distances and angles
    Rover.nav_dists = distance #rover_centric_pixel_distances
    Rover.nav_angles = angles#rover_centric_angles
    
    
    
    return Rover