import numpy as np
import cv2

def TemplateMatching(src, temp, stepsize): # src: source image, temp: template image, stepsize: the step size for sliding the template
    mean_t = 0;
    var_t = 0;
    location = [0, 0];
    # Calculate the mean and variance of template pixel values
    # ------------------ Put your code below ------------------ 
    mean_t,var_t=cv2.meanStdDev(temp);
    max_corr = 0;
    # Slide window in source image and find the maximum correlation
    for i in np.arange(0, src.shape[0] - temp.shape[0], stepsize):
        for j in np.arange(0, src.shape[1] - temp.shape[1], stepsize):
            mean_s = 0;
            var_s = 0;
            corr = 0;
            # Calculate the mean and variance of source image pixel values inside window
            # ------------------ Put your code below ------------------ 
            mean_s,var_s=cv2.meanStdDev(src);
            # Calculate normalized correlation coefficient (NCC) between source and template
            # ------------------ Put your code below ------------------ 
            a=0
            for x in range(1,temp.shape[0]):
                for y in range(1,temp.shape[1]):
                    a+=(temp[x][y]-mean_t)*(src[i+x][j+y]-mean_s)
            corr=(1/(temp.shape[0]*temp.shape[1]))*a/(var_t*var_s)
            #corr=np.array(range(src.shape[0]))@np.array(range(src.shape[1]))@(-mean_s)*(-mean_t)/(src.shape[0]*src.shape[1]*var_s*var_t)
            if corr > max_corr:
                max_corr = corr;
                location = [i, j];
    return location

# load source and template images
source_img = cv2.imread('source_img.jpg',0) # read image in grayscale
temp = cv2.imread('template_img.jpg',0)

# read image in grayscale
location = TemplateMatching(source_img, temp, 20);
print(location)
match_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

# Draw a red rectangle on match_img to show the template matching result
# ------------------ Put your code below ------------------ 
#match_img=cv2.cvtColor(source_img,cv2.COLOR_GRAY2RGB)
match_img=cv2.rectangle(match_img,(location[0],location[1]),(location[0]+temp.shape[1],location[1]+temp.shape[0]),(255,0,0),3)
# Save the template matching result image (match_img)
# ------------------ Put your code below ------------------ 
#cv2.write("asd.jpg",match_img)

# Display the template image and the matching result
cv2.namedWindow('TemplateImage', cv2.WINDOW_NORMAL)
cv2.namedWindow('MyTemplateMatching', cv2.WINDOW_NORMAL)
cv2.imshow('TemplateImage', temp)
cv2.imshow('MyTemplateMatching', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()