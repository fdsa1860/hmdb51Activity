% genTracklets

clear;clc;close all;

if ispc
    dataMainPath = 'D:\xikang\research\data\HMDB51';
else
    dataMainPath = '/home/xikang/research/data/HMDB51';
end

dataPath = fullfile(dataMainPath,'hmdb51_org');

annoPath = fullfile(dataMainPath,'AnnotationDense');
annoFile = dir(fullfile(annoPath,'*.bb'));
annoList = cell(1,length(annoFile));
for ai = 1:length(annoFile)
    annoList{ai} = annoFile(ai).name;
end
annoList = sort(annoList);

outputPath = fullfile(dataMainPath,'trackletOrig_gz');
if ~exist(outputPath,'dir')
    mkdir(outputPath)
end

videoList = {};
counter = 1;

childPath = dir(dataPath);

for ci = 1:length(childPath)
    
    if strcmp(childPath(ci).name,'.') || strcmp(childPath(ci).name,'..')
        continue;
    end
    
    actionPath = fullfile(dataPath, childPath(ci).name);
    
    video = dir(fullfile(actionPath,'*.avi'));
    for vi = 1:length(video)
        videoList{counter} = video(vi).name;
        videoPathList{counter} = actionPath;
        counter = counter + 1;
    end
end
[videoList,ind] = sort(videoList);
videoPathList = videoPathList(ind);

assert(length(videoList)==length(annoList));
clear sampleList;
sampleList(1:length(videoList)) = struct('name',[],'video',[],...
    'annotation',[]);
for si = 1:length(sampleList)
    [~,videoName,~] = fileparts(videoList{si});
    [~,annoName,~] = fileparts(annoList{si});
    assert(strcmp(videoName,annoName));
    sampleList(si).name = videoName;
    sampleList(si).video = fullfile(videoPathList{si}, videoList{si});
    sampleList(si).annotation = fullfile(annoPath, annoList{si});
end

gzFileList = [];
gzFile = dir(fullfile(outputPath,'*.gz'));
counter = 1;
for gi = 1:length(gzFile)
%     gzFileList{counter} = gzFile(gi).name;
    [~,temp,~] = fileparts(gzFile(gi).name);
    gzFileList{counter} = temp;
    counter = counter + 1;
end
gzFileList = sort(gzFileList);

for si = 1:length(sampleList)

    if any(ismember(gzFileList,sampleList(si).name))
        continue;
    end

    cmd = ['../3rdParty/improved_trajectory_original/release/DenseTrackStab ' sampleList(si).video ...
        ' -H ' sampleList(si).annotation ...
        ' | gzip -> ' outputPath '/' sampleList(si).name '.gz'];

    cmd = regexprep(cmd,'(','\\(');
    cmd = regexprep(cmd,')','\\)');
    cmd = regexprep(cmd,'&','\\&');
    cmd = regexprep(cmd,';','\\;');
    system(cmd);
    fprintf('%d of %d files are processed.\n',si,length(sampleList));
end

