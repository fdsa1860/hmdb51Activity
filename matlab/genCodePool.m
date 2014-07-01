% generate training set for clustering

% load('../expData/hmdb51_tracklet_fileSplit1.mat','trainSet', ...
%     'testSet', 'trainLabel', 'testLabel');
load('../expData/hmdb51_trackletOrig_fileSplit1.mat','trainSet', ...
    'testSet', 'trainLabel', 'testLabel');


step = 70;
k = 5;
X_size = 3000000;
% D = 30    % for only trajectories
D = 435;    % orginal densetrajectories
sampleSize = 100000;

rng(0); % initialize random seed;

X = zeros(X_size, D);
X_cnt = 0;
indices = [];
while X_cnt < X_size
    for i = 1:step:length(trainSet)
        ind = i + randi(step) - 1;
        while ismember(ind, indices)
            ind = i + randi(step) - 1;
        end
        indices = [indices ind];
        [~,file,tmpExt] = fileparts(trainSet{ind});
        if strcmp(tmpExt,'.gz')
            if ~isdir(fullfile('.','tmp'))
                mkdir(fullfile('.','tmp'));
            end
            gunzip( trainSet{ind},fullfile('.','tmp') )
            traj = load( fullfile('.','tmp',file) );
            delete( fullfile('.','tmp',file) );
        else
            traj = load(trainSet{ind});
        end
        L = size(traj,1);
        X_start = X_cnt + 1;
        X_end = min(X_cnt+L, X_size);
        X(X_start:X_end,:) = traj(1:X_end-X_start+1,2:end);
        X_cnt = X_end;
        if X_cnt == X_size
            break;
        end
    end
end

sampleInd = randi(X_size,sampleSize,1);
X = X(sampleInd,:);
X = X';
save CodePool_20140420 X;
rmdir( fullfile('.','tmp') );