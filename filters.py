import cv2
import numpy as np
import streamlit as st


# Refer to the application notebook implement the following filters

@st.cache_data
def bw_filter(img):
    # Write your code here to convert img to a gray image
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img_gray

@st.cache_data
def vignette(img, level=2):
	# Write your code here to create the vignette effect
	height, width = img.shape[:2]
	X_result_kernel = cv2.getGaussianKernel(width, width/level)
	Y_result_kernel = cv2.getGaussianKernel(height, height/level)
	kernel = Y_result_kernel * X_result_kernel.T
	mask = kernel / kernel.max()
	img_vignette = np.copy(img)
	for i in range(3):
		img_vignette[:, :, i] = img[:, :, i] * mask
	return img_vignette

@st.cache_data
def sepia(img):
	# Write your code here to create the sepia effect
	img_sepia = img.copy()
	img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
	img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189], 
													[0.349, 0.686, 0.168], 
													[0.272, 0.534, 0.131]]))

	# Clip values to the range [0, 255].
	img_sepia = np.clip(img_sepia, 0, 255)
	img_sepia = np.array(img_sepia, dtype = np.uint8)
	img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)

	return img_sepia

@st.cache_data
def pencil_sketch(img, ksize=5):
	# Write your code here to create the pencil sketch effect
	img_blur = cv2.GaussianBlur(img, (ksize,ksize), 0, 0)
	img_sketch, _ = cv2.pencilSketch(img_blur)
	
	return img_sketch

@st.cache_data
def color_sketch(img,ksize=7,sig=3):
	# invert the image
	inverted = 255 - img

	# Apply Gaussian blur
	gaussBlur = cv2.GaussianBlur(inverted, (ksize,ksize), 3, 3)

	# Invert the blurred image
	gaussBlur = 255 - gaussBlur

	# Apply the Gaussian blur to the original image
	colorSketch = cv2.divide(img, gaussBlur, scale = 256.0)

	return colorSketch

@st.cache_data
def photoSaturation(img,sat):
	# saturate the image
	saturated = cv2.addWeighted(img, sat, np.zeros(img.shape, img.dtype), 0, 0)
	return saturated

@st.cache_data
def oilPainting(img):
	# oil painting effect
	oilPainted = cv2.xphoto.oilPainting(img, 7, 1)
	return oilPainted

@st.cache_data
def cartoonize(img):
	# cartoonize the image
	cartoonized = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
	return cartoonized

@st.cache_data
def watercolor(img):
	# watercolor effect
	watercolored = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
	return watercolored

@st.cache_data
def hsv(img):
	# hsv effect
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	return hsv

@st.cache_data
def edge(img):
	#convert image to grayscale
	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#cleaning up the image using Gaussian blur
	img_gray_blur=cv2.GaussianBlur(img_gray,(5,5),0)

	#extract edges

	canny_edges=cv2.Canny(img_gray_blur,10,70)

	#do an invert binarize the image
	ret, mask=cv2.threshold(canny_edges,70,255,cv2.THRESH_BINARY_INV)
	fg = cv2.bitwise_or(img, img, mask=mask)
	mask = cv2.bitwise_not(mask)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

	return mask