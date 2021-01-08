# seamCarving
This is my (in privat collaboration with friends) implementation of the seamCarving algorithm based on this lecture: https://youtu.be/rpB6zQNsbQU.
The algorithm allows users to use content based scaling along the horizonral axis by removing unimportant pixels lines (seams) first.
The algorithm uses edge detection to figure out which areas of the image are important by ensuring that seams that go through edges are removed last.
Therefore the best case scenario for this algorithm is an image with a clear subject in the foreground and a clear background with no edges.

!IMPORTANT! The implementation is not optimized yet. Scaling the test image (salvadore_deli.png) by 50% takes about 20 minutes.

