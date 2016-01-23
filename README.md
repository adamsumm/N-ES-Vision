NOTE: This is heavily based on Matthew Guzdial's work which can be found at https://github.com/mguzdial3/VideoParser .

# VideoParser
Have you thought to yourself: 'There's a lot of gameplay video out there. I (a games researcher) would sure like to get at all the info in that game in a way that is maybe not perfect but gets the job done'? Well then this is the repo for you! 

# What's in Here
This repo contains three python scripts to help extract information on sprite usage per frame of a video. It also contains a sprite palette and gameplay video for a game that is almost believably not a clone of Super Mario Bros. to be used in the below tutorial. 

# Tutorial: Introduction
In this tutorial I'll go through the following steps: 
  1. Where to find python 2.7 and the relevant libraries. 
  2. How to use the "VideoParser.py" python script to extract information from video
  3. How to use the "ResizeSprites.py" python script to resize the sprite palette to better match the video.
  4. How to use the "VisualizeFrames.py" python script to check the extracted information

# Tutorial 1: Relevant Libraries/Downloads
Python 2.7: https://www.python.org/downloads/
Install pip: http://pip.readthedocs.org/en/stable/installing/
Install numpy: run "pip install numpy" in terminal
Install pillow: run "pip install pillow" in terminal
Install pyaml: run "pip install pyyaml" in terminal
Install OpenCV: 
	Mac: https://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support/
	Windows (not verified): http://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html#gsc.tab=0
Download ffmpeg (video parser): https://www.ffmpeg.org/download.html

#Tutorial 2: VideoParser.py
With all the relevant downloads downloaded, download the repo and cd into it. 

The VideoParser.py script uses the following command: 

python VideoParser.py <name of gameplay video> <name of sprites directory> <frames per second> <scaling of video>

Note: If you are running Windows, you'll need to comment line 120 and uncomment line 121

So to start with the Moorio example let's run the following command in the terminal: python VideoParser.py Moorio.mp4 moorioSprites 28 0.4484

#Tutorial 3: VideoTracker.py

At this point you'll have a file called frames/frameDescriptionMoorio.mp4_28_0.75.csv - The naming conventions goes frames/frameDescription<name of gameplay video>_<frames per second>_<image recognition threshold>.csv .  This can now be fed to a tracker that will form tracks trying to keep track of different objects, regardless if they move or change image.  

This is run by 

#Tutorial 4: VisualizeFrames.py

VisualizeFrames.py is a script to visualize a frameDescriptions.csv file to see how well the VideoParser.py did. The command to use it looks like: 

python VisualizeFrames.py <spritesDirectory> <frames csv file> <the directory to place the visualizations>

So in this case let's run the command: 

python VisualizeFrames.py moorioSprites/resizedSprites frames/frameDescriptions.csv visualizedCV

You should see a new directory "visualizedCV" that contains the visualizations of frameDescriptions.csv. These are pretty good! In the last section of this ReadMe I'll point out a few possible ways to up the output, but this is about as good as you'll get.

And that's it! Following that same set of steps you should be able to get a frameDescriptions.csv file. From there you can use the sprites and their positions per frame to extract all sorts of information about a gameplay video. 

#Tips and Tricks
Here's a couple additional suggestions for you: 
	-OpenCV cannot handle transparency. Your best bet is to include a background color on all sprites that either matches the background color of the game, have multiple different versions of each sprite with different background colors are limited, or create sprites with random noise as the background image. 

	-VideoParser.py has a number of ways to accept more or less possible sprites in a frame. The easiest way to do this is to adjust the "threshold" variable of the "FrameParser" class. The lower the value, the more (less likely) sprites will be found. However there are many more ways commented on throughout the file. 

	-Finding the optimal allocation of sprites on the screen is a tricky (NP-Hard) problem.  While there are often not many sprites on screen (ignoring bullet hell SHMUPs) it can still explode into a massive combinatorics problem.  As such there is a simulated annealing step going on for each frame.  This should probably be a part of the tracker (i.e. use past information to help assign new information), but isn't yet
