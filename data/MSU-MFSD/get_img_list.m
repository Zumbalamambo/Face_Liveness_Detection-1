clc; clear all; close all;

%% Create image file list
% list for positive images
pos_img_path = fullfile('.', 'cropped', 'positive');
img_list = dir(pos_img_path);
fileID = fopen(fullfile('.', 'pos_img_list.txt'), 'w');
for i = 3 : length(img_list)
    img_name = img_list(i).name;
    img_path = fullfile(pos_img_path, img_name);
    fprintf(fileID, '%s\n', img_path);
end
fclose(fileID);

% list for negative images
neg_img_path = fullfile('.', 'cropped', 'negative');
img_list = dir(neg_img_path);
fileID = fopen(fullfile('.', 'neg_img_list.txt'), 'w');
for i = 3 : length(img_list)
    img_name = img_list(i).name;
    img_path = fullfile(neg_img_path, img_name);
    fprintf(fileID, '%s\n', img_path);
end
fclose(fileID);

% list for all images
pos_reader = fopen('pos_img_list.txt', 'r');
neg_reader = fopen('neg_img_list.txt', 'r');
all_img_writer = fopen('test_images.txt', 'w');
all_label_writer = fopen('test_labels.txt', 'w');

neg_face_cnt = 0;
while feof(neg_reader) == 0
    line = fgetl(neg_reader);
    neg_face_cnt = neg_face_cnt + 1;
    fprintf(all_img_writer, '%s\n', line);
    fprintf(all_label_writer, '%d\n', 0);
end
fclose(neg_reader);

pos_face_cnt = 0;
while feof(pos_reader) == 0
    line = fgetl(pos_reader);
    pos_face_cnt = pos_face_cnt + 1;
    fprintf(all_img_writer, '%s\n', line);
    fprintf(all_label_writer, '%d\n', 1);
    if (pos_face_cnt == neg_face_cnt)
        % remove the bias of dataset
        break;
    end
end
fclose(pos_reader);
fclose(all_img_writer);
fclose(all_label_writer);

%% Create train/test/val set

% read in all image paths and labels
pos_img_files = cell(pos_face_cnt, 1);
neg_img_files = cell(neg_face_cnt, 1);
all_img_files = cell(pos_face_cnt + neg_face_cnt, 1);
all_labels = zeros(pos_face_cnt + neg_face_cnt, 1);
all_img_reader = fopen('all_img_list.txt', 'r');
all_label_reader = fopen('all_label_list.txt', 'r');

ind = 0;
while feof(all_img_reader) == 0
    image = fgetl(all_img_reader);
    label = fgetl(all_label_reader);
    ind = ind + 1;
    all_img_files{ind} = image;
    all_labels(ind) = str2double(label);
end
fclose(all_img_reader);
fclose(all_label_reader);

% shuffle and split the dataset
rng(0);
shuffle_ind = randperm(length(all_labels));
all_img_files = all_img_files(shuffle_ind);
all_labels = all_labels(shuffle_ind);

% get test samples
num_test_samples = ceil(length(all_labels) / 5);
test_img_files = all_img_files(1:num_test_samples);
test_labels = all_labels(1:num_test_samples);
all_img_files(1:num_test_samples) = [];
all_labels(1:num_test_samples) = [];

% get val samples
num_val_samples = ceil(length(all_labels) / 5);
val_img_files = all_img_files(1:num_val_samples);
val_labels = all_labels(1:num_val_samples);
all_img_files(1:num_val_samples) = [];
all_labels(1:num_val_samples) = [];

% get train samples
train_img_files = all_img_files;
train_labels = all_labels;

%% Write list files for different sets
test_img_writer = fopen('test_images.txt', 'w');
test_label_writer = fopen('test_labels.txt', 'w');
for i = 1 : length(test_img_files)
    fprintf(test_img_writer, '%s\n', test_img_files{i});
    fprintf(test_label_writer, '%d\n', test_labels(i));
end
fclose(test_img_writer);
fclose(test_label_writer);

val_img_writer = fopen('val_images.txt', 'w');
val_label_writer = fopen('val_labels.txt', 'w');
for i = 1 : length(val_img_files)
    fprintf(val_img_writer, '%s\n', val_img_files{i});
    fprintf(val_label_writer, '%d\n', val_labels(i));
end
fclose(val_img_writer);
fclose(val_label_writer);

train_img_writer = fopen('train_images.txt', 'w');
train_label_writer = fopen('train_labels.txt', 'w');
for i = 1 : length(train_img_files)
    fprintf(train_img_writer, '%s\n', train_img_files{i});
    fprintf(train_label_writer, '%d\n', train_labels(i));
end
fclose(train_img_writer);
fclose(train_label_writer);
