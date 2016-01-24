# import the necessary packages
import sys, os, cv, cv2, math, glob, csv
import numpy as np
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image, ImageOps
import random
import yaml
class FrameParser: 
	def __init__(self, threshold=0.8):
		self.threshold = threshold

	#Returns a list of columns with a description of the sprites in the passed in frame Image
	def GetFrameDescription(self, frameNumber, frame, sprites,matching, overlapThreshold ,Tmin,Tmax,iterations,metaIterations, minDist = 4):		
		columns = []

		for sprite,spritename in zip(sprites,spriteNames):
			#Pattern matching right here
			

			result = cv2.matchTemplate(sprite[0], frame, cv2.TM_CCOEFF_NORMED)#Change cv2.TM_CCOEFF_NORMED for more accurate but slower results
			(h, w) = sprite[0].shape[:2]
			#Get only the max values (those above the threshold) that represent a local maximum
			maxVals  = MaxLocalVals(result, w, h, self.threshold)
			for pt in maxVals:
				#Whether or not to add this to the list of columns
				addIt = True
				#The potential new column to add
				column = [str(frameNumber), sprite[1], str(pt[0]), str(pt[1]), str(w), str(h), str(result[pt[1]][pt[0]])]
				#The list of columns that this one forces out
				toRemove = []

				#Iterate through all current columns and ensure that no column subsumes this one and that this one subsumes no others
				for c in columns:
					#print column,c,matching["./"+spritesDirectory+"/"+c[1]]["./"+spritesDirectory+"/"+column[1]],OverlapAmount(c, column)
					#Feel free to add this back in, though it'll slow things down a bit it's good for when some sprites contain others
					#print column, c, matching["./"+spritesDirectory+"/"+c[1]]["./"+spritesDirectory+"/"+column[1]],OverlapAmount(c, column)
					'''
					if matching["./"+spritesDirectory+"/"+c[1]]["./"+spritesDirectory+"/"+column[1]] > 0.06:
						#Check if this column is inside one of the others
						
						if not c in toRemove:
							if OverlapAmount(c, column)>=overlapThreshold:
								print 'removing',c,column,OverlapAmount(c, column)
								# c is in column, remove c
								cScore = float(c[6])
								if cScore<result[pt[1]][pt[0]]:
									toRemove.append(c)
								else:
									addIt = False
									break 
					'''
					#Ensure that there isn't something better in this region that has a higher score
					if addIt and not c in toRemove:
						if abs(int(c[2])-int(column[2]))<=minDist and abs(int(c[3]) - int(column[3]) )<=minDist:
							c6 = float(c[6])
							if c6<result[pt[1]][pt[0]]:
								toRemove.append(c)
							else:
								addIt = False
								break

				#Removes those that need removing
				for r in toRemove:
					columns.remove(r)

				#Add those that need adding
				if addIt:
					columns.append(column)

			inconsistent = {tuple(c):set() for c in columns}
			for c1 in columns:
				c1 = tuple(c1)
				for c2 in columns:
					c2 = tuple(c2)
					if c1 != c2 and OverlapAmount(c1, c2)>=overlapThreshold and matching["./"+spritesDirectory+"/"+c1[1]]["./"+spritesDirectory+"/"+c2[1]] > 0.1:
						if c2 not in inconsistent[c1]:
							inconsistent[c1].add(c2)
						if c1 not in inconsistent[c2]:
							inconsistent[c2].add(c1)
			#print inconsistent
		bestSet = []
		bestScore = float('-inf')
		def neighbor(c):
			n = {cc for cc in c}
			rando = random.choice(list(n))
			n.remove(rando)
			for ic in inconsistent[rando]:
				if len(inconsistent[ic].intersection(n)) == 0:
					n.add(ic)
			return n
		chosen = set()
		potentials = [tuple(c) for c in columns]

		while len(potentials) > 0:
			c = random.choice(potentials)
			chosen.add(c)
			potentials.remove(c)
			for cO in inconsistent[c]:
				if cO in potentials:
					potentials.remove(cO)
		
		TFactor = -math.log(Tmax / Tmin)
		def getEnergy(s):
			energy = 0
			for c in s:
				energy += float(c[4])*float(c[5])*float(c[6])
			return frame.shape[0]*frame.shape[1]-energy
		
		ec = getEnergy(chosen)
		bestEnergy = ec
		bestSet = chosen
		if len(chosen) > 0:
			for ii in range(metaIterations):
				ec = bestEnergy
				chosen = bestSet
				print ii
				for ti in range(iterations):
					n = neighbor(chosen)
					en = getEnergy(n)
					T = Tmax*math.exp(TFactor*ti/iterations)
					dE = en-ec
					if  dE > 0 and  math.exp(-dE / T) < random.random():
						pass #min(math.exp(min(T*0.5,-dE)/T),1) < random.random():#(dE > 0 and math.exp(-dE/T)  < random.random()):
					else: 
						chosen = n
						ec = en
						if ec < bestEnergy:
							print 'newbest',ec
							bestEnergy = ec
							bestSet = chosen
			print frame.shape[0]*frame.shape[1]-bestEnergy

		columns = bestSet

		return columns


#Find the highest scoring top left corner within a range (in this case half width and half height)
def MaxLocalVals(result, width, height, threshold):
	loc = np.where( result >= threshold)
	locPoints = []


	for pt in zip(*loc[::-1]):
		maxVal = result[pt[1]][pt[0]]
		#New point to potentially add
		maxPnt = pt
		toRemove = []

		#Ensure that you have the best local points
		for pt2 in locPoints:
			if maxVal<result[pt2[1]][pt2[0]] and (abs(pt2[0]-pt[0])<width/2.0 and abs(pt2[1]-pt[1])<height/2.0):#Is this point already in locPoints and in range and is it better?
				maxPnt = pt2
			elif maxPnt==pt and (abs(pt2[0]-pt[0])<width/2.0 and abs(pt2[1]-pt[1])<height/2.0):#Inverse, Does this point beat out something in locPoints?
				toRemove.append(pt2)

		for pt2 in toRemove:
			locPoints.remove(pt2)
		if maxPnt== pt:
			locPoints.append(pt)

	return locPoints

#Check the amount of overlapping pixels between these two sprites, represented as columns
def OverlapAmount(column1, column2, printIt = False):

	#Rectangle defined by column1
	r1left = int(column1[2])
	r1right = r1left + int(column1[4])
	r1bottom = int(column1[3])
	r1top = r1bottom + int(column1[5])

	#Rectangle defined by column2
	r2left = int(column2[2])
	r2right = r2left + int(column2[4])
	r2bottom = int(column2[3])
	r2top = r2bottom + int(column2[5])

	left = max(r1left, r2left);
	right = min(r1right, r2right);
	top = min(r1top, r2top);
	bottom = max(r1bottom, r2bottom);
	SI = max(0,right-left)*max(0,top-bottom)
	SU = int(column1[4])*int(column1[5])+int(column2[4])*int(column2[5])-SI

	#SI = max(0, max(l1[0], l2[0]) -min(r1[0], r2[0]))*max(0, max(l1[1], l2[1]) -min(r1[1], r2[1]))
	# width1*height1 + width2*height2
	#SU = ((r1[0]-l1[0])*(r1[1]-l1[1]))+((r2[0]-l2[0])*(r2[1]-l2[1]))-SI
	if SU==0:
		return 0
	#print SI, SU
	return float(SI) / float(SU)
	

if __name__ == '__main__':
	FFMPEG_BIN = "ffmpeg" # Use on Linux ans Mac OS
	#FFMPEG_BIN = "ffmpeg.exe" # Use on Windows

	import yaml
	inputfile = sys.argv[1]
	with open(inputfile,'rb') as f:
		inputfile = yaml.load(f)


	#Example call: python VideoParser.py Gameplay.mp4 sprites 1
	video = inputfile['video']
	spritesDirectory = inputfile['spritesDirectory']
	framesPerSecond = inputfile['framesPerSecond']
	threshold = inputfile['imageThreshold']
	#The folder that the frame images will end up in
	folder = inputfile['outputFolder']	
	if not os.path.exists(folder):
		os.makedirs(folder)
	
	scale = inputfile['videoScaling']

	overlapThreshold = inputfile['overlapThreshold']
	Tmin = inputfile['Tmin']
	Tmax = inputfile['Tmax']
	iterations = inputfile['iterations']
	metaIterations = inputfile['metaIterations']
	#Run the parser to generate the frame images
	fname = folder+"frameDescriptions{}_{}_{}.csv".format(video,framesPerSecond,threshold)
	if not os.path.isfile(fname):
		os.system(FFMPEG_BIN+ " -i "+video+" -r {}".format(framesPerSecond)+' -vf scale=iw*{}'.format(scale)  +':ih*{}'.format(scale)  + ' ' +folder+"image-%08d.png")#" -vf scale="+widthOfFrame+":"+heightOfFrame+
		
		sprites = []
		spriteNames ={}
		directory = "./"+spritesDirectory+"/"
		for filename in glob.glob(directory+"*.png"):
			spriteNames[filename] = ImageOps.grayscale(Image.open(filename))
			img_rgb = cv2.imread(filename)
			sprite_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)#Gray is faster to match
			splits = filename.split("/")
			spritename = splits[len(splits)-1]
			sprites.append((sprite_gray, spritename))

		#Initialize Frame Parser
		fp = FrameParser(threshold)

		target = open(fname,"wb")
		writer = csv.writer(target)
		column = ["frame", "spritename", "x", "y", "w", "h", "confidence"]
		writer.writerow(column)

		gaussian_kernel_sigma = 1.5
		gaussian_kernel_width = 11
		gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

		matching = {}
		maxVal = float('-inf')
		for sprite1 in spriteNames:
			matching[sprite1]  = {}
			for sprite2 in spriteNames:
				matching[sprite1][sprite2] = SSIM(spriteNames[sprite1], gaussian_kernel_1d).ssim_value(spriteNames[sprite2])#np.max(sp.signal.correlate2d(sprites[sprite1],sprites[sprite2]))
				if matching[sprite1][sprite2] > maxVal:
					maxVal = matching[sprite1][sprite2]

		for s1 in matching:
			for s2 in matching[s1]:
				matching[s1][s2] /= maxVal
				print s1, s2, matching[s1][s2]
		for frameFile in glob.glob(folder+"*.png"):
			print "Frames: "+str(frameFile)
			frame = cv2.imread(frameFile)
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			splits =frameFile.split("image-")
			frameNumber = int(splits[len(splits)-1][:-4])

			columns = fp.GetFrameDescription(frameNumber,frame_gray,sprites,matching, overlapThreshold ,Tmin,Tmax,iterations,metaIterations)

			for c in columns:
				writer.writerow(c)
