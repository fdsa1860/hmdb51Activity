% unzip tracklets data

clear;clc;close all;

if ispc
    trackletPath = 'D:\xikang\research\data\HMDB51\tracklet_gz';
    outputPath = 'D:\xikang\research\data\HMDB51\tracklet';
else
    trackletPath = '/home/xikang/research/data/HMDB51/tracklet_gz';
    outputPath = '/home/xikang/research/data/HMDB51/tracklet';
end

gunzip(fullfile(trackletPath,'*.gz'), outputPath);