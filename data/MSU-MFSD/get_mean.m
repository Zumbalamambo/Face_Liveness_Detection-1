clc; clear all; close all;

%% For train set
sum_im = zeros(224, 224, 3);
train_reader = fopen('train_images.txt', 'r');
cnt = 0;
while feof(train_reader) == 0
    im_name = fgetl(train_reader);
    im = im2double(imread(im_name));
    sum_im = sum_im + im;
    cnt = cnt + 1;
    if mod(cnt, 100) == 0
        fprintf('cnt: %d\n', cnt);
    end
end
fclose(train_reader);

sum_im = double(sum_im) / cnt;
train_mean_R = mean(mean(sum_im(:, :, 1))) * 255;
train_mean_G = mean(mean(sum_im(:, :, 2))) * 255;
train_mean_B = mean(mean(sum_im(:, :, 3))) * 255;

fprintf('The mean value of each channel (BGR) is (%f, %f, %f)', train_mean_B, train_mean_G, train_mean_R);
% (110.266284, 125.741553, 155.540794)