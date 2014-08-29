% read split file
% Output: a cell array containing tracklet file names

clear;clc;close all;

% splitPath = '/home/xikang/research/data/HMDB51/testTrainMulti_7030_splits';
splitPath = '/Users/xikang/Documents/data/HMDB51/testTrainMulti_7030_splits';
splitFiles1 = dir(fullfile(splitPath,'*split1.txt'));
splitFiles2 = dir(fullfile(splitPath,'*split2.txt'));
splitFiles3 = dir(fullfile(splitPath,'*split3.txt'));

trackletPath = '/Users/xikang/Documents/data/HMDB51/trackletOrig';
% trackletPath = '/home/xikang/research/data/HMDB51/trackletOrig_gz';

splitFiles = splitFiles1;

trainSet = cell(70*length(splitFiles),1);
testSet = cell(30*length(splitFiles),1);
trainLabel = zeros(70*length(splitFiles),1);
testLabel = zeros(30*length(splitFiles),1);

allDataSet = cell(150*length(splitFiles),1);
allDataLabel = zeros(150*length(splitFiles),1);
allDataSplit = zeros(150*length(splitFiles),1);
allDataCount = 1;

for si = 1:length(splitFiles)
    fid = fopen(fullfile(splitPath,splitFiles(si).name));
    cFile = textscan(fid,'%s%f');
    trainSet(70*(si-1)+1:70*si) = cFile{1}(cFile{2}==1);
    testSet(30*(si-1)+1:30*si) = cFile{1}(cFile{2}==2);
    trainLabel(70*(si-1)+1:70*si) = si;
    testLabel(30*(si-1)+1:30*si) = si;
    indBegin = allDataCount;
    indEnd = allDataCount + length(cFile{2}) - 1;
    allDataSet(indBegin:indEnd) = cFile{1};
    allDataLabel(indBegin:indEnd) = si;
    allDataSplit(indBegin:indEnd) = cFile{2};
    allDataCount = indEnd + 1;
    fclose(fid);
end

emptyCells = cellfun(@isempty,allDataSet);
allDataLabel(emptyCells) = [];
allDataSplit(emptyCells) = [];
allDataSet(emptyCells) = [];

% add path and modify the extension name
for j = 1:length(trainSet)
    [~,tempName,~] = fileparts(trainSet{j});
%     trainSet{j} = fullfile(trackletPath,[tempName '.gz']);
    trainSet{j} = fullfile(trackletPath,tempName);
end
for j = 1:length(testSet)
    [~,tempName,~] = fileparts(testSet{j});
%     testSet{j} = fullfile(trackletPath,[tempName '.gz']);
    testSet{j} = fullfile(trackletPath,tempName);
end
for j = 1:length(allDataSet)
    [~,tempName,~] = fileparts(allDataSet{j});
%     allDataSet{j} = fullfile(trackletPath,[tempName '.gz']);
    allDataSet{j} = fullfile(trackletPath,tempName);
end

save hmdb51_trackletOrig_fileSplit1 trainSet testSet trainLabel ...
    testLabel allDataSet allDataLabel allDataSplit;