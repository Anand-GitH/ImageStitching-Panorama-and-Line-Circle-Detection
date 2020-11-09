import cv2
import numpy as np

def Hline_accum(img,p_res=1,theta_res=1):
    #Hough Line Accumulator is voting based algorithm 
    #We transform the image to the parameter space for line it just two parameters 
    #xcos(theta) + ysin(theta) = p
    #Increment the cell if point which has the same p and theta values
    #the one with more votes(more pixels) is the line with theta and p 
    
    h,w=img.shape
    
    #Calculate image diagonal using the pythagoras theorem
    diag=np.ceil(np.sqrt(h**2+w**2))
    
    #creating a parameter space with axis as p and theta
    ps= np.arange(-diag, diag + 1, p_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    
    #Hough Accumulator - A matrix to track the votes
    Hough  = np.zeros((len(ps), len(thetas)), dtype=np.uint64)
    xs, ys = np.nonzero(img)   
    
    
    #Find each edge point and accumulate votes
    for e in range(len(xs)):
        x=xs[e]
        y=ys[e]
        
        for t in range(len(thetas)):
            p = int((x * np.cos(thetas[t]) + y * np.sin(thetas[t])) + diag)
    
            Hough[p,t] +=1 #voting
        
    return Hough,ps,thetas

def getHLines(H, nhlines, thresh=0):
    #This method allows to select the lines with more number of votes if threshold parameter is 0
    #if threshold parameter has value then it selects only lines less than that threshold
    
    Hbackup = np.copy(H)
    nlinesidx = []
    
    for i in range(nhlines):
        idx = np.argmax(Hbackup)
        line_idx = np.unravel_index(idx, Hbackup.shape) 
        votes=Hbackup[line_idx]
        
        if thresh != 0:
            while votes>thresh:
                Hbackup[line_idx]=0
                idx = np.argmax(Hbackup)
                line_idx = np.unravel_index(idx, Hbackup.shape) 
                votes=Hbackup[line_idx]
        
        Hbackup[line_idx]=0
        nlinesidx.append(line_idx)
        
    return nlinesidx


def drawHLines(img, hLines, p, thetas, color,ignoreidx,filename):
    
    with open (filename, 'w') as filehandler:
        for i in range(len(hLines)):
            if len(ignoreidx)!=0:
                if i not in ignoreidx:  
                    rho = p[hLines[i][0]]
                    theta = thetas[hLines[i][1]]

                    filehandler.write(str([np.degrees(theta),rho])+'\n')
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho

                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(img, (y1, x1), (y2, x2), color, 2)

            else:
                rho = p[hLines[i][0]]
                theta = thetas[hLines[i][1]]

                filehandler.write(str([np.degrees(theta),rho])+'\n')

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho

                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img, (y1, x1), (y2, x2), color, 5)
            
def removerlaps(linelist,H,mindist=0):
    final_list=[]    
    distance=[]

    for i in range(len(linelist)):
        dist=[]
        for j in range(len(linelist)):
            if i == j:
                dist.append(99999)  #set max distance for the same node 
            else:
                dist.append(cv2.norm(np.array(linelist[i]),np.array(linelist[j]), normType=cv2.NORM_L2))
                            
        
        distance.append([dist[np.argmin(dist)],np.argmin(dist)])
    
    for i in range(len(distance)):
        if distance[i][0]<mindist:
            if H[linelist[i]]>H[linelist[distance[i][1]]]:
                final_list.append(linelist[i])
            else:
                final_list.append(linelist[distance[i][1]])
        else:
            final_list.append(linelist[i])
    
    final_list=np.unique(final_list,axis=0).tolist()
    return final_list
        
#Read Image and convert into gray scale
srcimg = cv2.imread('Hough.png')
gimg=cv2.cvtColor(srcimg,cv2.COLOR_BGR2GRAY)


##################################################Draw Circles########################################################
cimg = cv2.Canny(gimg, 50, 100, None, 3, False)
circles = cv2.HoughCircles(cimg.T,cv2.HOUGH_GRADIENT,2,20,param1=500,param2=100,minRadius=18,maxRadius=60)

circles = np.uint16(np.around(circles))
with open ('results/coins.txt', 'w') as filehandler:
    for i in circles[0,:]:
        filehandler.write(str([i[0],i[1],i[2]])+'\n')
        cv2.circle(srcimg,(i[1],i[0]),i[2],(0,255,0),4)
        
cv2.imwrite("results/coins.jpg", srcimg)
#################################################Line Processing#####################################################
bsrcimg = cv2.imread('Hough.png')
gimg=  cv2.cvtColor(bsrcimg, cv2.COLOR_RGB2GRAY)
bimg = cv2.GaussianBlur(gimg, (5, 5), 1.5)
cimg = cv2.Canny(bimg, 100, 200)


houghmat,p,thetas=Hline_accum(cimg)


#######Get Hough Lines based on the length of the lines(votes)
hlines  = getHLines(houghmat, 30)
hlines1 = getHLines(houghmat, 1,148)
hlines2 = getHLines(houghmat, 8,78)
hlines3 = getHLines(houghmat, 127, 50)

hlines.extend(hlines1)
hlines.extend(hlines2)
hlines.extend([hlines3[-1]])

#################################################Cross Lines#########################################################
filteredcl=[]
bsrcimg = cv2.imread('Hough.png')

for i in range(len(hlines)):
        theta = thetas[hlines[i][1]]
        
        if theta>-0.96 and theta<-0.92:
            filteredcl.append(hlines[i])

filteredcl=removerlaps(filteredcl,houghmat,mindist=10)
drawHLines(bsrcimg, filteredcl, p, thetas, (203,149,54),[5,7,9],'results/blue_lines.txt')           
cv2.imwrite("results/blue_lines.jpg", bsrcimg)

#################################################Vertical Lines#########################################################
filteredvl=[]
bsrcimg = cv2.imread('Hough.png')

for i in range(len(hlines)):
        theta = thetas[hlines[i][1]]
        if theta>-1.54 and theta<-1.51:
            filteredvl.append(hlines[i])

filteredvl=removerlaps(filteredvl,houghmat,mindist=10)

drawHLines(bsrcimg, filteredvl, p, thetas, (11,11,191), [5,7,8], 'results/red_lines.txt')
cv2.imwrite("results/red_lines.jpg", bsrcimg)
#######################################################################################################################