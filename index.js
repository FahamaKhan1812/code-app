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

// ---------------------- LAB 05 ----------------------	
rgb = imread('cancer.jpg');
gray = rgb2gray(rgb); % convert rgb to gray
gray = histeq(gray); % gray level normalization using equalization

bin_thresh = logical(zeros(size(gray))); % create empty array for binary matrix

for r = 1: size(gray,1)
    
    for c= 1: size(gray,2)
         
        if(gray(r,c)<100)
            
            bin_thresh(r,c) = 1; % binary thresholding to set 1
            
        else
           
            bin_thresh(r,c) = 0; % binary thresholding to set 0
            
        end  
    end
end
    
b = imclearborder(bin_thresh,4); % remove some background noise

se = strel('disk',1);
c = imerode(b,se); % apply erosion to shrink obj and also to remove tiny noise
se = strel('disk',4);
c = imdilate(c,se); % apply dilation to expand obj
c = imfill(c,'holes');

rp = regionprops(c, 'BoundingBox', 'Area'); % create rectangle to identify objects

area = [rp.Area].';
[area,ind] = sort(area,'descend'); % sort area on the basis of largest obj in image
bb = rp(ind(1)).BoundingBox;
bb2 = rp(ind(2)).BoundingBox;

d = zeros(size(c,1),size(c,2),3); % fill color represent lung


for row = 1:size(c,1)
   for col = 1:size(c,2)
      if(c(row,col) == 1)
         
          d(row,col,1) = 255;
          d(row,col,2) = 255;
          d(row,col,3) = 255;
          
      end
   end
    
end
figure,imshowpair(gray,d,'montage');
rectangle('Position', bb, 'EdgeColor', 'red'); % plot red color rectangles
rectangle('Position', bb2, 'EdgeColor', 'red');

bw=im2bw(d,.2);
% bw = bwareaopen(bw,5000,8);
A = regionprops(bw,'Area');
a=bwarea(bw); % area of lungs
p=bwperim(bw,8); % perimeter of lungs
figure,imshow(p); % print parimeter of lungs image
imshow(rgb(:,:,1).*uint8(bw))
%figure,imshow(d);
impixelinfo;


// ---------------------- LAB 06 ----------------------	
image = imread('Z:\jump.jpg'); % read image
% get image dimensions: an RGB image has three planes
% reshaping puts the RGB layers next to each other generating
% a two dimensional grayscale image
[height, width, planes] = size(image);

rgb = reshape(image, height, width * planes);
imagesc(rgb); % visualize RGB planes

r = image(:, :, 1); % red channel
g = image(:, :, 2); % green channel
b = image(:, :, 3); % blue channel
 
threshold = 100; % threshold value
imagesc(b < threshold); %display the binarized image
 
% apply the blueness calculation

blueness = double(b) - max(double(r), double(g));
imagesc(blueness); % visualize RGB planes

mask = blueness < 45;
imagesc(mask);
labels = bwlabel(mask);
id = labels(200, 200); 
man = (labels == id);
imagesc(man); 
% save the image in PPM (portable pixel map) format
imwrite(man, 'Z:\man.ppm');
imshow(man);


// ---------------------- LAB 07 (Character Identification) ----------------------	
I = imread('C:/Users/19b-037-cs/Desktop/DIP-Lab-07-SP23/training.png');
imshow(I);
Igray = rgb2gray(I); 
%imshow(Igray);
 
Ibw = imbinarize(Igray,graythresh(Igray));
%imshow(Ibw)
Iedge = edge(uint8(Ibw));
%imshow(Iedge)
 
se = strel('square',2);
Iedge2 = imdilate(Iedge, se);
%Iedge2 = imerode(Iedge,se);
%imshow(Iedge2);
Ifill= imfill(Iedge2,'holes');
%imshow(Ifill);
 
[Ilabel, num] = bwlabel(Ifill);
disp(num);
Iprops = regionprops(Ilabel);
Ibox = [Iprops.BoundingBox];
Ibox = reshape(Ibox,[4 50]);
%imshow(I);
 
hold on;
for cnt = 1:50
rectangle('position',Ibox(:,cnt),'edgecolor','r');
end

// ---------------------- LAB 08 (Laplacian Filters Technique) ----------------------	
// 1
% Read the image
 
image = imread('C:/Users/19b-037-cs/Desktop/DIP-Lab-08-SP23/moon.png');
% Create the Laplacian filter kernel
laplacianFilter = fspecial('laplacian');
 
% Apply the Laplacian filter to the image
filteredImage = imfilter(image, laplacianFilter);
 
image2= image-filteredImage;
 
% Display the original and filtered images
figure;
subplot(1,3,1), imshow(image), title('Original Image');
subplot(1,3,2), imshow(filteredImage), title('Laplacian Image');
subplot(1,3,3), imshow(image2), title('Result');

// 2
originalImage = imread('C:/Users/19b-037-cs/Desktop/DIP-Lab-08-SP23/skeleton.png');
% Read the image
 
% Display the original image
figure('Name', 'Image Processing Steps');
subplot(2, 3, 1);
imshow(originalImage);
title('Original Image');
 
% a Apply Laplacian filter
laplacianFilter = fspecial('laplacian');
laplacianImage = imfilter(originalImage, laplacianFilter, 'replicate');
subplot(2, 3, 2);
imshow(laplacianImage, []);
title('Laplacian Filtered Image');
 
% b Subtract Laplacian filter to display sharpened image
sharpenedImage = originalImage - laplacianImage;
subplot(2, 3, 3);
imshow(sharpenedImage);
title('Sharpened Image');
 
% c Apply Sobel filter
sobelFilter = fspecial('sobel');
sobelImage = imfilter(originalImage, sobelFilter, 'replicate');
subplot(2, 3, 4);
imshow(sobelImage);
title('Sobel Filtered Image');
 
% d Smooth image by a 5x5 averaging filter
averageFilter = fspecial('average', [5 5]);
smoothedImage = imfilter(originalImage, averageFilter, 'replicate');
subplot(2, 3, 5);
imshow(smoothedImage);
title('Smoothed Image');
 
% e Apply power law to smoothed image
gamma = 0.5; 
powerLawImage = imadjust(smoothedImage, [], [], gamma);
subplot(2, 3, 6);
imshow(powerLawImage);
title('Power Law Image');

// ---------------------- LAB 09 (Cascade Object Detector) ----------------------	
faceDetector = vision.CascadeObjectDetector;
I = imread('C:/Users/19b-037-cs/Desktop/DIP-Lab-09-SP23/team.jpg');
bboxes = step(faceDetector, I);
IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, 'Face');
figure, imshow(IFaces), title('Detected faces');

// ---------------------- LAB 10 (detecting-a-cell-using-image-segmentation) ----------------------	
I = imread('cell.tif');
[~,threshold] = edge(I,'sobel');
fudgeFactor = 0.5;
BWs = edge(I,'sobel',threshold * fudgeFactor);

se90 = strel('line',3,90);
se0 = strel('line',3,0);
BWsdil = imdilate(BWs,[se90 se0]);

BWdfill = imfill(BWsdil,'holes');
BWnobord = imclearborder(BWdfill,4);


seD = strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);

BWoutline = bwperim(BWfinal);
Segout = I; 
Segout(BWoutline) = 255; 

figure;
subplot(2,4,1),imshow(I),title('Original Image');
subplot(2,4,2),imshow(BWs),title('Binary Gradient Mask');
subplot(2,4,3),imshow(BWsdil),title('Dilated Gradient Mask');
subplot(2,4,4),imshow(BWdfill),title('Binary Image with Filled Holes');
subplot(2,4,5), imshow(BWnobord),title('Cleared Border Image');
subplot(2,4,6), imshow(BWfinal),title('Segmented Image');
subplot(2,4,7), imshow(labeloverlay(I,BWfinal)),title('Mask Over Original Image')
subplot(2,4,8), imshow(Segout),title('Outlined Original Image')

         `
      );
    },
  });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
