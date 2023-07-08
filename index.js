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
  
		
         `
      );
    },
  });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
