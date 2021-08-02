import sys, os
from PIL import Image

# change paths and file names to fit needs
infilename = 'det_val.json'
indirname = 'data/bdd100k/labels/det_20/'

labelsfile = open(indirname + infilename, 'r')
line = labelsfile.readline()
outdirname = 'data/bdd100kmodded/validation/label_2/'
outfilenum = 0
imgsindirname = 'data/bdd100k/val/'
imgsoutdirname = 'data/bdd100kmodded/validation/image_2/'

# read until end of file
while line != '':
    index = line.find('name')
    
    # found name, look for other attributes
    if index != -1:
        infilename = line[index+8:index+25]

        # repeatedly look for characteristics of objects
        line = labelsfile.readline()
        lines = []
        occluded, truncated, x1, y1, x2, y2 = None, None, None, None, None, None

        # look until end of "labels" attribute
        while line.find(']') == -1:
            # category of object; disregard if not Pedestrian
            if line.find('"category":') != -1:
                if line[line.find('"category":')+13:line.find('",')] != 'pedestrian':
                    # object is not a pedestrian; disregard information
                    occluded, truncated, x1, y1, x2, y2 = None, None, None, None, None, None

            # occluded/truncated information, convert boolean to int/float
            elif line.find('occluded') != -1:
                occluded = 1 if line[line.find('occluded')+12] == 't' else 0
            elif line.find('truncated') != -1:
                truncated = 1.0 if line[line.find('truncated')+13] == 't' else 0.0

            # bbox 2d information
            elif line.find('"x1":') != -1:
                x1 = float(line[line.find('x1')+5:line.find(',')])
            elif line.find('"y1":') != -1:
                y1 = float(line[line.find('y1')+5:line.find(',')])
            elif line.find('"x2":') != -1:
                x2 = float(line[line.find('x2')+5:line.find(',')])
            elif line.find('"y2":') != -1:
                y2 = float(line[line.find('y2')+5:line.find('\n')])
            
            # found new object, record data if any
            elif line.find('"id":') != -1 and occluded is not None:
                # putting in 0 for irrelevant 3d values
                output = 'Pedestrian %.1f %d 0 %.2f %.2f %.2f %.2f 0 0 0 0 0 0 0\n' % (truncated, occluded, x1, y1, x2, y2)
                lines.append(output)

            line = labelsfile.readline()
        
        # write to file if any pedestrians found
        if lines != []:
            try:
                # use pillow to copy file and convert properly
                img = Image.open(imgsindirname + infilename + '.jpg')
                img.save(imgsoutdirname + '%06d.png' % outfilenum)

                outfile = open(outdirname + '%06d.txt' % outfilenum, 'w')
                for l in lines:
                    outfile.write(l)
                outfile.close()

                print('saved %06d' % outfilenum)
                outfilenum += 1

            except FileNotFoundError:
                print('file does not exist in 100k set')

            outfile = open(outdirname + '%06d.txt' % outfilenum, 'w')
            for l in lines:
                outfile.write(l)
            outfile.close()

            print('saved %06d' % outfilenum)
            outfilenum += 1

    line = labelsfile.readline()

labelsfile.close()
