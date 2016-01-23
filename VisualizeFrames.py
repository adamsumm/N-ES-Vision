import csv, sys, os, copy, glob, cv2
sys.path.append('/usr/local/lib/python2.7/site-packages')
from PIL import Image
def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points
class FrameObject:
	def __init__(self, name, positionX, positionY, width, height):
		self.name = name
		self.x = positionX
		self.y = positionY
		self.width = width
		self.height = height

if __name__ == '__main__':
	spriteDirectory = sys.argv[1]
	frameCSV = sys.argv[2]
	translatedFramesDirectory = "./"+sys.argv[3]
	velocityFile = None
	if len(sys.argv) == 5:
		velocityFile = sys.argv[4]
	if not os.path.exists(translatedFramesDirectory):
		os.makedirs(translatedFramesDirectory)

	#STEP 1: Grab the sprites
	sprites = {}

	directory = "./"+spriteDirectory+"/"
	for filename in glob.glob(directory+"*.png"):
		img_rgb = Image.open(filename)
		keep = img_rgb.copy()

		splits = filename.split("/")
		spritename = splits[len(splits)-1]
		sprites[spritename] = keep
		img_rgb.close()#Ensure that we don't hit python's "too many files open" error

	#STEP 2: This section grabs the frame objects per frame as just strings
	source = open(frameCSV,"rb")
	reader = csv.reader(source)

	readLine = False#skip the first frame
	currFrame = 0
	prevFrame = 0
	gameplayVals = {}

	for row in reader:
		if readLine:
			currFrame = int(row[0])
			if not currFrame==prevFrame:
				gameplayVals.setdefault(currFrame, [])
				prevFrame = currFrame
			
			objName = str(row[1])
			gameplayVals[currFrame].append(FrameObject(objName,int(row[2]),int(row[3]), int(row[4]), int(row[5])))
			prevFrame = currFrame

		readLine=True
	velocities = {}
	if velocityFile:
		with open(velocityFile) as velFile:
			for line in velFile:
				dat = line.rstrip().split(',')
			
				if int(dat[0]) not in velocities:
					velocities[int(dat[0])] = []
				dat = [int(float(d)) for d in dat]
				velocities[int(dat[0])].append(dat)

	#STEP 3: Print the frames
	for translatedFrame in gameplayVals.keys():
		print "Translating Frame: "+str(translatedFrame)
		filenameStr = translatedFramesDirectory+"/"+str(translatedFrame)+".png"

		#Find max points of image to use as height and width
		maxX = 0
		maxY = 0
		for frameObject in gameplayVals[translatedFrame]:
			if frameObject.x+frameObject.width>maxX:
				maxX = frameObject.x+frameObject.width
			if frameObject.y+frameObject.height>maxY:
				maxY = frameObject.y+frameObject.height


		#Make the frame image
		img = Image.new('RGB', (int(maxX), int(maxY)), "rgb(91, 147, 251)")#CHANGE THIS TO CHANGE THE DEFAULT BACKGROUND COLOR
		pixels = img.load()
		#Iterate through every object in this frame and draw it
		for o in gameplayVals[translatedFrame]:
			if o.name in sprites.keys():
				sprite = sprites[o.name]
				pix = sprite.load()
				#print 'pasting ', o.name , '@', o.x,o.y
				#img.paste(sprite,(int(o.x),int(o.y)))
				
				(width, height) = sprite.size
				for x in range(0, int(width)):
					for y in range(0, int(height)):
						xIndex = int(o.x)+x
						yIndex = int(o.y)+y
						r = int((pix[x,y][0]))
						g = int((pix[x,y][1]))
						b = int((pix[x,y][2]))
						newVal = (r,g,b)
						
						pixels[xIndex,yIndex]=newVal
		if velocityFile:
			if translatedFrame in velocities:
				for vels in velocities[translatedFrame]:
					print vels
					line = get_line((int(vels[1]), int(vels[2])),(int(vels[1])+int(vels[3]),int(vels[2])+int(vels[4])))
					for pt in line:
						if pt[0] >= 0 and pt[1] >= 0 and pt[0] < maxX and pt[1] < maxY:
							pixels[pt[0],pt[1]] = (255,255,255)

				
		img.save(filenameStr)
