const express = require("express");

const app = express();
const port = 3000;

app.get("/help", (req, res) => {
  res.format({
    text: function () {
      res.send(
        `	
	// --------------------------- Sobel Filter ----------------------
 	% Read the input image
inputImage = imread('img1.jpg');

% Convert the image to grayscale
grayImage = rgb2gray(inputImage);

% Apply Laplacian filter
laplacianImage = edge(grayImage, 'log');

% Apply Sobel filter
sobelImage = edge(grayImage, 'sobel');

% Apply Canny filter
cannyImage = edge(grayImage, 'canny');

% Display the original and filtered images side by side
figure;
subplot(2, 2, 1);
imshow(inputImage);
title('Original Image');

subplot(2, 2, 2);
imshow(laplacianImage);
title('Laplacian Filter');

subplot(2, 2, 3);
imshow(sobelImage);
title('Sobel Filter');

subplot(2, 2, 4);
imshow(cannyImage);
title('Canny Filter');

	// ---------------------- Power transform ,Logarithmic transform, Negation transform ----------------------
% Load the image
img = imread('image.jpg');
% Negation transform
neg_img = 255 - img;
% Logarithmic transform
log_img = uint8(255 * log(1 + double(img))/log(1 + double(max(max(img)))));
% Power transform
gamma = 1.2;
pow_img = uint8(255 * (double(img)/255).^gamma);
% Display the original and transformed images
figure, subplot(2,2,1), imshow(img), title('Original Image');
subplot(2,2,2), imshow(neg_img), title('Negation Transform');
subplot(2,2,3), imshow(log_img), title('Logarithmic Transform');
subplot(2,2,4), imshow(pow_img), title('Power Transform');

	// --------------------LAB No 04 ----------------------
	// ---------------------- image_averaging ----------------------

  % Image averaging for noise reduction

clear, clc, close all

% Load test image
img = im2double(rgb2gray(imread('G:\Courses\Digital Image Processing\Lecture Slides-Wasim\New Lectures-Stanford\3. Combining Images\Image Averaging for Noise Reduction\quadnight.jpg')));

% Loop over number of samples
nArray = [1, 2, 8, 32];
for i = 1 : numel(nArray)
    N = nArray(i);
    avgImg = zeros(size(img));
    
    % Add Gaussian noise, accumulate average image
    for j = 1 : N
        noisyImg = imnoise(img, 'gaussian', 0, 0.02);
        avgImg = avgImg + noisyImg;
    end % end j
    
    % Show image
    avgImg = avgImg / N;
    subplot(2, 2, i);
    imshow(avgImg);
    title([num2str(N), ' image(s)']);
    imwrite(avgImg, ['Image_Averaging_', num2str(N), 'avg.png']);
end % end i

// ---------------------- color_alignment ----------------------

% Take three pictures, each representing the R, G, B color 
% channels and align them to form the colored picture
% name of the input file
imname = 'C:\Users\Hussain Computer\Desktop\part1_1.jpg';
% read in the image
fullim = imread(imname);
% convert to double matrix (might want to do this later on to same memory)
fullim = im2double(fullim);
% compute the height of each part (just 1/3 of total)
height = floor(size(fullim,1)/3);
% separate color channels
B = fullim(1:height,:);
G = fullim(height+1:height*2,:);
R = fullim(height*2+1:height*3,:);
% Align the images
% Functions that might be useful to you for aligning the images include: 
% "circshift", "sum", and "imresize" (for multiscale)
% imshow(R)
% imshow(G)
% imshow(G)
% aR = align(R,B);
% aG = align(G,B);
RGB = cat(3, R, G, B);
imshow(RGB)
% ssd = sum(sum((R -G) .^2));
% display(ssd);
% open figure
%% figure(1);
% create a color image (3D array)
% ... use the "cat" command
% show the resulting image
% ... use the "imshow" command
% save result image
%% imwrite(colorim,['result-' imname]);


// ---------------------- im_defect ----------------------

% Load test images 
origImg =  im2double(imread('G:\Courses\Digital Image Processing\Lecture Slides-Wasim\New Lectures-Stanford\3. Combining Images\Defect Detection\pcbCropped.png'));
defectImg = im2double(imread('G:\Courses\Digital Image Processing\Lecture Slides-Wasim\New Lectures-Stanford\3. Combining Images\Defect Detection\pcbCroppedTranslatedDefected.png')); 

%(10, 10) shifted
% Perform shift
[row, col] = size(origImg);
xShift = 10;
yShift = 10;
registImg = zeros(size(defectImg));
registImg(yShift + 1 : row, xShift + 1 : col) = defectImg(1 : row - yShift, 1 : col - xShift);

% Show difference images
diffImg1 = abs(origImg - defectImg);
subplot(1, 3, 1), imshow(diffImg1); 
title('Unaligned Difference Image');

diffImg2 = abs(origImg - registImg);
subplot(1, 3, 2), imshow(diffImg2); 
title('Aligned Difference Image');

bwImg = diffImg2 > 0.15;
[height, width] = size(bwImg);
border = round(0.05*width);
borderMask = zeros(height, width);
borderMask(border:height-border, border:width-border) = 1;

bwImg = bwImg .* borderMask;
subplot(1, 3, 3), imshow(bwImg); 
title('Thresholded + Aligned Difference Image');

% Save images
imwrite(diffImg1, 'Defect_Detection_diff.png');
imwrite(diffImg2, 'Defect_Detection_diffRegisted.png');
imwrite(bwImg, 'Defect_Detection_bw.png');

// ---------------------- im_subtraction ----------------------
% Image subtraction
clear, clc, close all

% Load test images
maskImg = im2double(imread('mask.jpg'));
liveImg = im2double(imread('live.jpg'));

% Calculate difference image and enhance contrast
diffImg = abs(maskImg - liveImg);
histeqDiffImg = adapthisteq(diffImg, 'ClipLimit', 0.005);

% Show images
subplot(1, 4, 1), imshow(liveImg);
title('Live image');
subplot(1, 4, 2), imshow(maskImg);
title('Mask image');
subplot(1, 4, 3), imshow(diffImg);
title('Difference image');
subplot(1, 4, 4), imshow(histeqDiffImg);
title('Histogram equalized difference image');

% Save images
imwrite(diffImg, 'Image_Subtraction_diff.png');
imwrite(histeqDiffImg, 'Image_Subtraction_histeqdiff.png');

// ---------------------- im_mask_difference ----------------------
% Image subtraction example from IC manufacturing:
% die-to-die comparison of photomasks

clear, clc, close all;

% Load test images
maskImg1 = im2double(imread('G:\Courses\Digital Image Processing\Lecture Slides-Wasim\New Lectures-Stanford\3. Combining Images\Inspection of Photomask\mask1.png'));
maskImg2 = im2double(imread('G:\Courses\Digital Image Processing\Lecture Slides-Wasim\New Lectures-Stanford\3. Combining Images\Inspection of Photomask\mask2.png'));

% Perform subtraction
diffImg = abs(maskImg1 - maskImg2);
imshow(diffImg, []); 
title('Difference Image');
imwrite(diffImg, 'Mask_Comparison_diff.png')

// ---------------------- Dilation erosion using structure element ----------------------
A = imread('lungcancer.PNG');

imhist(A)
I = imgaussfilt(A,0.1)
k=imadjust(I,[],[],1.5)

level=graythresh(k)
bw=imbinarize(rgb2gray(k),level)

SE = strel('rectangle',[40 30]);

BW3 = imerode(bw,SE);
imshow(BW3)

BW4 = imdilate(BW3,SE);
imshow(BW4)
		
         `
      );
    },
  });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
