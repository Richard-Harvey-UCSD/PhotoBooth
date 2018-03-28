%% Part 1
%% Exercise 1:
clear all; close all; clc;

f = 2000;   % frequency of sinusoid
a = 1.5;    % amplitude of waveform
w = 2*pi*f; % angular frequency

fs = 0.5*w;   % sample sinusoid at 0.5*w
dt = 1/fs;
t = 0:dt:0.001;  

y = a*sin(w*t);
subplot(231);
plot(t,y);

xlabel('Time');
ylabel('Amplitude');
title('2kHz Sinusoid Sampled at 0.5*w');

fs = 1*w;   % sample sinusoid at 1*w
dt = 1/fs;
t = 0:dt:0.001;  

y = a*sin(w*t);
subplot(232);
plot(t,y);

xlabel('Time');
ylabel('Amplitude');
title('2kHz Sinusoid Sampled at 1*w');

fs = 2*w;   % sample sinusoid at 2*w
dt = 1/fs;
t = 0:dt:0.001;  

y = a*sin(w*t);
subplot(233);
plot(t,y);

xlabel('Time');
ylabel('Amplitude');
title('2kHz Sinusoid Sampled at 2*w');

fs = 4*w;   % sample sinusoid at 4*w
dt = 1/fs;
t = 0:dt:0.001;  

y = a*sin(w*t);
subplot(234);
plot(t,y);

xlabel('Time');
ylabel('Amplitude');
title('2kHz Sinusoid Sampled at 4*w');

fs = 5*w;   % sample sinusoid at 5*w
dt = 1/fs;
t = 0:dt:0.001;  

y = a*sin(w*t);
subplot(235);
plot(t,y);

xlabel('Time');
ylabel('Amplitude');
title('2kHz Sinusoid Sampled at 5*w');

%% Exercise 2:
clear all; close all; clc;

[audio fs] = audioread('audio.wav');    % read in audio 
%sound(audio,fs)

leftChannel = audio(:,1);
rightChannel = audio(:,2);
t = 0:1/fs:(length(leftChannel)-1)/fs;  % rescale time axis to seconds

plot(t,leftChannel);   % plot first channel
hold on;
plot(t,rightChannel);   % plot second channel

%axis('tight');
axis([0 21 -0.5 0.5]);  % set x-min/max and y-min/max axes 
xlabel('Time (sec)');   % set x axis label
ylabel('Amplitude');    % set y axis label
title('Audio Signal');  % add title to plot
legend('left channel','right channel'); % add legend

%% Exercise 3:
clear all; close all; clc;

[audio fs] = audioread('audio.wav');    % read in audio 

t0 = 0:1/fs:(length(audio(:,1))-1)/fs;
% plot(t0,audio(:,1);
% hold on;
% plot(t0,audio(:,2);
% sound(audio,fs);
% pause

fn = 1000;  % resample rate
[p,q] = rat(fn/fs);
newAudio1 = resample(audio,p,q);    % audio resampled at 1kHz
t1 = 0:1/fn:(length(newAudio1(:,1))-1)/fn;  % rescale time axis to seconds

% sound(newAudio1,fn);
% pause

subplot(221)
plot(t1,newAudio1(:,1));   % plot first channel
hold on;
plot(t1,newAudio1(:,2));   % plot second channel

axis([0 21 -0.5 0.5]);  % set x-min/max and y-min/max axes 
xlabel('Time (sec)');   % set x axis label
ylabel('Amplitude');    % set y axis label
title('Audio Signal Resampled at 1kHz');  % add title to plot
legend('left channel','right channel'); % add legend

fn = 1500;  % resample rate
[p,q] = rat(fn/fs);
newAudio2 = resample(audio,p,q);    % audio resampled at 1.5kHz
t2 = 0:1/fn:(length(newAudio2(:,1))-1)/fn;  % rescale time axis to seconds

% sound(newAudio2,fn);
% pause

subplot(222)
plot(t2,newAudio2(:,1));   % plot first channel
hold on;
plot(t2,newAudio2(:,2));   % plot second channel

axis([0 21 -0.5 0.5]);  % set x-min/max and y-min/max axes 
xlabel('Time (sec)');   % set x axis label
ylabel('Amplitude');    % set y axis label
title('Audio Signal Resampled at 1.5kHz');  % add title to plot
legend('left channel','right channel'); % add legend

fn = 10000;  % resample rate
[p,q] = rat(fn/fs);
newAudio3 = resample(audio,p,q);    % audio resampled at 10kHz
t3 = 0:1/fn:(length(newAudio3(:,1))-1)/fn;  % rescale time axis to seconds

% sound(newAudio3,fn);
% pause

subplot(223)
plot(t3,newAudio3(:,1));   % plot first channel
hold on;
plot(t3,newAudio3(:,2));   % plot second channel

axis([0 21 -0.5 0.5]);  % set x-min/max and y-min/max axes 
xlabel('Time (sec)');   % set x axis label
ylabel('Amplitude');    % set y axis label
title('Audio Signal Resampled at 10kHz');  % add title to plot
legend('left channel','right channel'); % add legend

fn = 44000;  % resample rate
[p,q] = rat(fn/fs);
newAudio4 = resample(audio,p,q);    % audio resampled at 44kHz
t4 = 0:1/fn:(length(newAudio4(:,1))-1)/fn;  % rescale time axis to seconds

% sound(newAudio4,fn);
% pause

subplot(224)
plot(t4,newAudio4(:,1));   % plot first channel
hold on;
plot(t4,newAudio4(:,2));   % plot second channel

axis([0 21 -0.5 0.5]);  % set x-min/max and y-min/max axes 
xlabel('Time (sec)');   % set x axis label
ylabel('Amplitude');    % set y axis label
title('Audio Signal Resampled at 44kHz');  % add title to plot
legend('left channel','right channel'); % add legend

%% Exercise 4:
close all; clc;

% sampling frequncies
Fn0 = 44100;
Fn1 = 1000;
Fn2 = 1500;
Fn3 = 10000;
Fn4 = 44000;

% time values from challenge 3
L0 = length(t0);    
L1 = length(t1);
L2 = length(t2);
L3 = length(t3);
L4 = length(t4);

%Compute the Fourier Transform
Y0 = fft(audio);        % original signal
Y1 = fft(newAudio1);    % resampled signal at 1kHz  
Y2 = fft(newAudio2);    % resampled signal at 1.5kHz  
Y3 = fft(newAudio3);    % resampled signal at 10kHz  
Y4 = fft(newAudio4);    % resampled signal at 44kHz  

%Compute and plot the single sided PSD 
P00 = abs(Y0/L0);
P0 = P00(1:L0/2+1);
P0(2:end-1) = 2*P0(2:end-1);
f0 = Fn0*(0:(L0/2))/L0;

P11 = abs(Y1/L1);
P1 = P11(1:L1/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f1 = Fn1*(0:(L1/2))/L1;

P22 = abs(Y2/L2);
P2 = P22(1:L2/2+1);
P2(2:end-1) = 2*P2(2:end-1);
f2 = Fn2*(0:(L2/2))/L2;

P33 = abs(Y3/L3);
P3 = P33(1:L3/2+1);
P3(2:end-1) = 2*P3(2:end-1);
f3 = Fn3*(0:(L3/2))/L3;

P44 = abs(Y4/L4);
P4 = P44(1:L4/2+1);
P4(2:end-1) = 2*P4(2:end-1);
f4 = Fn4*(0:(L4/2))/L4;

% plot original audio power spectrum
figure(1)
plot(f0,P0)
title('Single-Sided Amplitude Spectrum of Original Audio with Fs = 44.1kHz') 
xlabel('f (Hz)')
ylabel('|P0(f)|')

% plot resampled audio power spectrums
figure(2)
subplot(221)
plot(f1,P1)
title('Single-Sided Amplitude Spectrum of Resampled Audio at 1kHz') 
xlabel('f (Hz)')
ylabel('|P1(f)|')

subplot(222)
plot(f2,P2)
title('Single-Sided Amplitude Spectrum of Resampled Audio at 1.5kHz') 
xlabel('f (Hz)')
ylabel('|P2(f)|')

subplot(223)
plot(f3,P3)
title('Single-Sided Amplitude Spectrum of Resampled Audio at 10kHz') 
xlabel('f (Hz)')
ylabel('|P3(f)|')

subplot(224)
plot(f4,P4)
title('Single-Sided Amplitude Spectrum of Resampled Audio at 44kHz') 
xlabel('f (Hz)')
ylabel('|P4(f)|')

%% MatLab + Arduino Extra Credit Challenge
% Refer to MatlabGUI.m for modifications
clear all; close all; clc;

run MatlabGUI.m

%% Part 2
%% Exercise 1a:
%Black and white checkerboard
clear all; close all; clc;

A = [255 0 255 0 255 0;
     0 255 0 255 0 255;
     255 0 255 0 255 0;
     0 255 0 255 0 255;
     255 0 255 0 255 0;
     0 255 0 255 0 255;];
A = uint8(A);
figure(1)
imshow(A)
truesize(gcf,[100 100])
set (gcf, 'Units', 'normalized', 'Position', [0.1,0.1,0.4,0.7]); 
title('Checkerboard Image from Matrix A')

%% Exercise 1b:
% German Flag
clear all; clc; close all;

A_R = [zeros(2,6); 255*ones(4,6)]; 
A_G = [zeros(2,6); zeros(2,6); 255*ones(2,6)]; 
A_B = [zeros(6,6)];
A = zeros(6,6,3); 
A(:,:,1) = A_R; 
A(:,:,2) = A_G; 
A(:,:,3) = A_B;
imshow(uint8(A))
truesize(gcf,[100 100])
set (gcf, 'Units', 'normalized', 'Position', [0.1,0.1,0.7,0.7]);
title('German Flag')

%% Exercise 2a:
clear all; clc; close all;

IMG = imread('Navy.jpg');
R = IMG(:,:,1);
G = IMG(:,:,2);
B = IMG(:,:,3);

a = zeros(size(IMG,1),size(IMG,2));
just_R = cat(3,R,a,a);
just_G = cat(3,a,G,a);
just_B = cat(3,a,a,B);

subplot(221);
imshow(IMG);
title('Original Image');

subplot(222);
imshow(just_R);
title('Red Layer of Original Image');

subplot(223);
imshow(just_G);
title('Green Layer of Original Image');

subplot(224);
imshow(just_B);
title('Blue Layer of Original Image');

% scale down most prevalent color layer
new_B = (0.2 .* B);
new_image = cat(3,R,G,new_B);
scaled_just_B = cat(3,a,a,new_B);
figure
imshow(scaled_just_B)
title('Blue Layer Scaled by Factor of 0.2');
figure
imshow(new_image)
title('Reconstructed Image with Scaled Down Blue Layer');

%% Exercise 2b:

a = zeros(size(IMG,1),size(IMG,2));
just_R = cat(3,R,a,a);
just_G = cat(3,a,G,a);
just_B = cat(3,a,a,B);

subplot(221);
imshow(IMG);
title('Original Image');

subplot(222);
imshow(just_R);
title('Red Layer of Original Image');

subplot(223);
imshow(just_G);
title('Green Layer of Original Image');

subplot(224);
imshow(just_B);
title('Blue Layer of Original Image');


%% Exercise 2c:

% scale down most prevalent color layer
new_B = (0.2 .* B);
new_image = cat(3,R,G,new_B);
scaled_just_B = cat(3,a,a,new_B);
figure
imshow(scaled_just_B)
title('Blue Layer Scaled by Factor of 0.2');
figure
imshow(new_image)
title('Reconstructed Image with Scaled Down Blue Layer');

%% Exercise 2d:
% Comment on why either image looks better, 
% i.e. original or reconstructed

%% Exercise 2e:
clear all; clc; close all;

IMG = imread('stained_teeth.jpg');
R = IMG(:,:,1);
G = IMG(:,:,2);
B = IMG(:,:,3);

% show original image for comparison
subplot(142);
imshow(IMG);
title('Original Image');

% whiten teeth
scaled_B = (3 .* B);
newImage = cat(3,R,G,scaled_B);
subplot(143);
imshow(newImage);
title('Properly Whitened Teeth');

% darken teeth
scaled_B = (.33 .* B);
newImage = cat(3,R,G,scaled_B);
subplot(141);
imshow(newImage);
title('Yellowed Teeth');

% over-whiten teeth
scaled_B = (10 .* B);
newImage = cat(3,R,G,scaled_B);
subplot(144)
imshow(newImage);
title('Overly Whitened Teeth');

%% Exercise 3a:
clear all; clc; close all;

IMG = imread('dark.jpg');
new_IMG = histeq(IMG);

% compare images
figure(1);
subplot(121);
imshow(IMG);
title('Original Image');
subplot(122);
imshow(new_IMG);
title('Histogram Equalized Image');

% compare histograms
figure(2);
subplot(121);
imhist(IMG);
title('Original Image');
subplot(122);
imhist(new_IMG);
title('Histogram Equalized Image');

%% Exercise 3b:
clear all; clc; close all;

IMG = imread('dark.jpg');
new_IMG = imadjust(IMG,stretchlim(IMG),[]);

% compare images
figure(1);
subplot(121);
imshow(IMG);
title('Original Image');
subplot(122);
imshow(new_IMG);
title('Contrast Stretched Image');

% compare histograms
figure(2);
subplot(121);
imhist(IMG);
title('Original Image');
subplot(122);
imhist(new_IMG);
title('Contrast Stretched Image');

%% Exercise 3c:
% Comment on comparing the results from Contrast Strectching and Histogram
% Equalization with regards to the original image.


%% Exercise 4a:
clear all; clc; close all;

IMG = imread('OutlineNavy.png');
imshow(IMG);
title('Image filtered with Outline Filter');

%% Exercise 4b:
clear all; clc; close all;

% step 1)
IMG = imread('Penguin.jpg');

% step 2)
figure(1);
subplot(121);
imshow(IMG);
subplot(122);
imhist(IMG);

% step 3)
new_IMG = medfilt2(IMG);
figure(2);
subplot(121);
imshow(new_IMG);
subplot(122);
imhist(new_IMG);

% step 4)
H = fspecial('average',3);
filtered_IMG = imfilter(new_IMG,H);

% step 5)
figure(3);
subplot(121);
imshow(filtered_IMG);
subplot(122);
imhist(filtered_IMG);

figure(4);
subplot(121);
imshow(IMG);
title('Original Image');
subplot(122);
imshow(filtered_IMG);
title('Original Image with Noise Removed');

%% Photobooth Challenge for ECE 5
% The purpose of this program is to take a photo from a webcam and do some 
% signal processing with the image. We will be manipulating the pixels by 
% various means such as changing the color or flipping the image.
% Resources - Much of the code can be found on the previous part of Lab 3 
% or http://blogs.mathworks.com/steve/category/special-effects/

% Clears figure windows, variables, commands, etc.
clear all; close all; clc;

% Shows what webcams are connected to the computer
webcamlist

% Creates webcam as an object and tests to see that webcam is working.
cam = webcam(1); % input maybe 2 or 3 if using attached webcam 
preview(cam);
% msgbox('Press enter to begin 3-second countdown','Welcome');
pause(3);

% Take first image
countdown = figure('Name','Countdown','NumberTitle','off',...
    'Color','black','MenuBar','none','ToolBar','none',...
    'Units','normalized','Position',[0.75 0.75 0.25 0.25]);
imshow('Photobooth_countdown_3.png','Border','tight');
pause(1);
imshow('Photobooth_countdown_2.png','Border','tight');
pause(1);
imshow('Photobooth_countdown_1.png','Border','tight');
pause(1);
close(countdown);

% save and display first image
%img = figure('Name','First Image','NumberTitle','off','Visible','off');
image1 = snapshot(cam);
img1 = imresize(image1,[480 640]);
imwrite(image1, 'image1.jpg'); 

% Creating a black and white image
x = rgb2gray(img1);
black_and_white = uint8(zeros(480, 640, 3));
% uint8() fixes the matrix size so it can go in photoreel
black_and_white(:,:,1) = x;
% stores the black and white image into different layers
black_and_white(:,:,2) = x; 
black_and_white(:,:,3) = x;
%imshow('image1.jpg'); 

% Take second image
countdown = figure('Name','Countdown','NumberTitle','off',...
    'Color','black','MenuBar','none','ToolBar','none',...
    'Units','normalized','Position',[0.75 0.75 0.25 0.25]);
imshow('Photobooth_countdown_3.png','Border','tight');
pause(1);
imshow('Photobooth_countdown_2.png','Border','tight');
pause(1);
imshow('Photobooth_countdown_1.png','Border','tight');
pause(1);
close(countdown);

% save and display second image
%img = figure('Name','Second Image','NumberTitle','off','Visible','off');
image2 = snapshot(cam);
img2 = imresize(image2,[480 640]);
imwrite(image2, 'image2.jpg'); 

% Decorrelation streching - useful for visulaizing image by reducing 
% inter-plane autocorelation levels in an image.
colorImg = decorrstretch(img2);
colorImg = imadjust(colorImg,[0.10; 0.79],[0.00; 1.00], 1.10);
%imshow('image2.jpg'); 

% Take third image
countdown = figure('Name','Countdown','NumberTitle','off',...
    'Color','black','MenuBar','none','ToolBar','none',...
    'Units','normalized','Position',[0.75 0.75 0.25 0.25]);
imshow('Photobooth_countdown_3.png','Border','tight');
pause(1);
imshow('Photobooth_countdown_2.png','Border','tight');
pause(1);
imshow('Photobooth_countdown_1.png','Border','tight');
pause(1);
close(countdown);

% save and display third image
%img = figure('Name','Third Image','NumberTitle','off','Visible','off');
image3 = snapshot(cam);
img3 = imresize(image3,[480 640]);
imwrite(image3, 'image3.jpg'); 

% Flipping an image
a = flipud(img3(:,:,1)); 
b = flipud(img3(:,:,2)); 
c = flipud(img3(:,:,3));
flippedImg = cat(3,a,b,c);
%imshow('image3.jpg'); 

% Closing windows
closePreview(cam); % closes the preview window 
delete(cam); % closes the webcam

image0 = imread('ECE5.png');
img0 = imresize(image0,[480 640]);

%compositeImg = [img0; img1; img2; img3];
compositeImg = [img0; black_and_white; colorImg; flippedImg];
imwrite(compositeImg, 'Photostrip.jpg');
imshow(compositeImg);

