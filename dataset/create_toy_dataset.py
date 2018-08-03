# Create a toy dataset for testing purposes
# https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html
#
# Authors:
# agray3 (Original version)
# luckymouse0 (Modified version)

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import random
import math

# number of images we are going to create in each of the two classes
nfigs=250

# number of images for validation and testing
nval = 0
ntest = 0

# Specify the image size. 
# E.g. size=32 will create images with 32x32 pixels
size=32


def create_dataset(nfigs, dir):
    #loop over classes
    for clss in ["squares","triangles"]:
        print ("generating images of "+clss+":")
        
        #loop over number of images to generate
        for i in range(nfigs):

            #initialise a new figure 
            fig, ax = plt.subplots()

            #initialise a new path to be used to draw on the figure
            Path = mpath.Path

            #set position and scale of each shape using random numbers
            #the coefficients are used to just try and prevent too many shapes from 
            #spilling off the edge of the image
            basex=0.7*random.random()
            basey=0.7*random.random()
            length=0.5*random.random()
            
            if clss == "squares":
                path_data= [
                    (Path.MOVETO, (basex, basey)), #move to base position of this image
                    (Path.LINETO, (basex+length, basey)), #draw line across to the right
                    (Path.LINETO, (basex+length, basey+length )), #draw line up
                    (Path.LINETO, (basex, basey+length)), #draw line back across to the left
                    (Path.LINETO, (basex, basey)), #draw line back down to base postiion
                ]
            else: #triangles
                path_data= [
                    (Path.MOVETO, (basex, basey)), #move to base position of this image
                    (Path.LINETO, (basex+length, basey)), #draw line across to the right
                    (Path.LINETO, ((basex+length/2.), 
                        basey+(math.sqrt(3.)*length/2.))), #draw line to top of equilateral triangle
                    (Path.LINETO, (basex, basey)), #draw line back to base position            
                ]

            #get the path data in the right format for plotting
            codes, verts = zip(*path_data)
            path = mpath.Path(verts, codes)

            #add shade the interior of the shape
            patch = mpatches.PathPatch(path, facecolor='gray', alpha=0.5)
            ax.add_patch(patch)
            
            #set the scale of the overlall plot
            plt.xlim([0,1])
            plt.ylim([0,1])

            #swith off plotting of the axis (only draw the shapes)
            plt.axis('off')

            #set the number of inches in each dimension to one
            # - we will control the number of pixels in the next command
            fig.set_size_inches(1, 1)

            # save the figure to file in te directory corresponding to its class
            # the dpi=size (dots per inch) part sets the overall number of pixels to the
            # desired value
            fig.savefig(dir+clss+'/data'+str(i)+'.png',dpi=size)   
            # close the figure
            plt.close(fig)
            
def main():
    create_dataset(nfigs, dir='train/')
    
    if(nval > 0):
        create_dataset(nval, dir='val/')
    
    if(ntest > 0):
        create_dataset(nval, dir='test/')
    
if __name__ == "__main__":
    main()
    