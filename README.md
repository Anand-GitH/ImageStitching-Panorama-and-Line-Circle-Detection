# ImageStitching Panorama and Line-Circle Detection
Panorama - Linear stitching
1. Identify the key features and feature descriptions using SIFT 
2. Calculate Similarity (distance) between these key features of two images
3. Apply Lowe's test to identify best matches
4. Calculate homograpy 
5. Warp Perpective and align images

Line-Circle Detection - Hough Transformation:
Hough Transformation is a voting based algorithm in which we create parameter space with p and theta 
and if the pixel point falls in that space then its incremented by 1 the one with more votes 
are the outstanding lines. So if the length of lines are small then it will have lesser votes as it contains less pixel points

1. Convert image to binary image
2. Create hough transform matrix with parameter range
3. Accumulate pixel points in the range
4. Search for lines with the preper threshold - number of votes
5. Apply angles to filter out wrong lines and circles
6. Remove overlaps with the higher voted lines 
7. plot on the image the identified lines and circles.

Note: Code here assumes the x axis is vertical and y axis - horizontal 
But cv2 hough transforms considers x-axis as horizontal and y axis as vertical.




