# Code for drone photogrammetry calculations 

"""
pixels_to_meters converts a dimension in pixels to a dimension in meters
inputs:
    - pixels: dimension in pixels
kwargs:
    - sensor width: camera sensor width in mm
    - img_width: 
    - altitude: drone altitude from water surface
    - depth: depth of shark
    - focal_length: camera focal length in mm
outputs: 
    -   dimension in m

Values for DJI Mavic 2 drone in Santa Elena Bay, CR: sensor_width=13.2, img_width=1920, altitude=37, depth=1.5, focal_length=28
"""
def pixels_to_meters(pixels, sensor_width=13.2, img_width=1920, altitude=37, depth=1.5, focal_length=28):
    altitude += depth
    pixel_dim = sensor_width / img_width
    pixel_dim /= 1000
    focal_length /= 1000
    GSD = (altitude / focal_length) * pixel_dim
    return GSD * pixels