% test fisher vector
% Xikang Zhang, 06/04/2014

% function testFisherFeat
clear;clc;close all;

nInitCenter = 256;

% % Load data
% file = 'CodePool_20140417.mat'; % 300000
% load(fullfile('../expData',file));
% X = X(136:243,:);
% Y = X';
% save('CodePool_L30_20140507.txt','Y','-ascii');

%% learn cluster centers
% addpath('../3rdParty/litekmeans');
% [~,trainCenter] = litekmeans(X,nInitCenter);
% rmpath('../3rdParty/litekmeans');

% tic
% [IDX, trainCenter] = kmeans(X', nInitCenter,'EmptyAction','singleton');
% toc

% cmd = '';
% cmd = [cmd '../3rdParty/kmlocal-1.7.2/bin/my_kmlsample'];
% cmd = [cmd ' -df ../expData/CodePool_L30_20140507.txt'];
% cmd = [cmd ' -d 30 -k 4000 -max 100000'];
% system(cmd);
% save kmeans_hmdb51_matlab_init4000_L108_20140514 trainCenter;
% load('../expData/kmeans_hmdb51_matlab_init4000_L108_20140514.mat','trainCenter');

% D = size(X,1);
% D2 = floor(D/2);
% tic
% [U,Score] = princomp(X');
% toc
% X2 = Score(:,1:D2)';
% tic;
% [means, covariances, priors] = vl_gmm(X2, nInitCenter);
% toc
% rmpath(genpath('../3rdParty/vlfeat-0.9.18/toolbox'));
% save gmm256_hmdb51_20140604 means covariances priors;
% load ../expData/gmm256_hmdb51_L108_20140604;

%% get dense trajectory BOW features
% load('../expData/hmdb51_trackletOrig_fileSplit1.mat','allDataSet', ...
%     'allDataLabel','allDataSplit');
% startTime = tic;
% if ~exist('tmp','dir')
%     mkdir(fullfile('.','tmp'));
% end
% 
% D2 = size(means,1);
% fFeat = zeros(2*nInitCenter*D2,length(allDataSet));
% % load dFeat6625;
% indToErase = false(size(allDataLabel));
% for i=1:length(allDataSet)
% % for i = 6626:length(allDataSet)
%     tic
%     [~,file,ext] = fileparts(allDataSet{i});
%     
%     if strcmp(ext,'.gz')
%         gunzip(allDataSet{i},fullfile('.','tmp'));
%         traj = load(fullfile('.','tmp',file));
%         delete(fullfile('.','tmp',file));
%     end
%     
%     if isempty(traj)
%         indToErase(i) = true;
%         continue;
%     end
%     
%     X = traj(:,137:137+108-1)';    % ignore the first 10 elements
%     [U,Score] = princomp(X');
%     X2 = Score(:,1:D2)';
%     class_hist = vl_fisher(X2, means, covariances, priors);
%     fFeat(:,i) = class_hist;
%     
%     save fFeat fFeat
%     fprintf('%d of %d files are processed.\n',i,length(allDataSet));
%     toc
% end
% 
% fFeat(:,indToErase) = [];
% allDataLabel(indToErase) = [];
% allDataSplit(indToErase) = [];
% 
% if exist('tmp','dir')
%     rmdir(fullfile('.','tmp'));
% end
% 
% toc(startTime)

% save fFeat256_hmdb51_L108_20140514 dFeat allDataLabel allDataSplit
load ../expData/fFeat256_hmdb51_L108_20140514.mat



% % normalization
% sum_hFeat_train = sum(hFeat_train,1);
% hFeat_train = hFeat_train./bsxfun(@times,sum_hFeat_train,ones(size(hFeat_train,1),1));
% sum_hFeat_test = sum(hFeat_test,1);
% hFeat_test = hFeat_test./bsxfun(@times,sum_hFeat_test,ones(size(hFeat_test,1),1));

%% train a SVM problem

X2_train = fFeat(:,allDataSplit==1);
y2_train = allDataLabel(allDataSplit==1)';
X2_test = fFeat(:,allDataSplit==2);
y2_test = allDataLabel(allDataSplit==2)';

% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));
% Cind = -10:10;
% G = 10.^Cind;
% C = 2.^Cind;
% % C = 512;
% accuracyMat = zeros(length(G),length(C));
% for gi = 1:length(G)
% for ci = 1:length(C)
%     ly = unique(y2_train);
%     svmModel = cell(1,length(ly));
%     accuracy = zeros(1,length(ly));
%     for i=1:length(ly)
%         y_train2 = y2_train;
%         y_train2(y2_train==ly(i)) = 1;
%         y_train2(y2_train~=ly(i)) = -1;
%         y_validate2 = y2_validate;
%         y_validate2(y2_validate==ly(i)) = 1;
%         y_validate2(y2_validate~=ly(i)) = -1;
%         y_test2 = y2_test;
%         y_test2(y2_test==ly(i))=1;
%         y_test2(y2_test~=ly(i))=-1;
%         model = svmtrain(y_train2',X2_train',sprintf('-s 0 -t 2 -c %d -g %d',C(ci),G(gi)));
%         [predict_label, ~, prob_estimates] = svmpredict(y_validate2', X2_validate', model);
%         accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
% %         [predict_label, ~, prob_estimates] = svmpredict(y2_test', X2_test', model);
% %         accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
%         svmModel{i} = model;
%     end
%     % accuracy
%     fprintf('\naccuracy is %f\n',mean(accuracy));
%     accuracyMat(ci,gi) = mean(accuracy);
% end
% end
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));

%% train a SVM problem using one versus all, liblinear
addpath(genpath('../3rdParty/liblinear-1.93/matlab'));
% Cind = -1:10;
% C = 2.^Cind;
C = 100;
accuracyMat = zeros(1,length(C));
for ci = 1:length(C)
%     ly = unique(y2_train);
%     svmModel = cell(1,length(ly));
%     accuracy = zeros(1,length(ly));
%     for i=1:length(ly)
        y_train2 = y2_train;
%         y_train2(y2_train==ly(i)) = 1;
%         y_train2(y2_train~=ly(i)) = -1;
%         y_validate2 = y2_validate;
%         y_validate2(y2_validate==ly(i)) = 1;
%         y_validate2(y2_validate~=ly(i)) = -1;
        y_test2 = y2_test;
%         y_test2(y2_test==ly(i))=1;
%         y_test2(y2_test~=ly(i))=-1;
        model = train(y_train2',sparse(X2_train'),sprintf('-s 2 -c %d',C(ci)));
%         [predict_label, ~, prob_estimates] = predict(y_validate2', sparse(X2_validate'), model);
%         accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
            [predict_label, ~, prob_estimates] = predict(y_test2', sparse(X2_test'), model);
            accuracy = nnz(predict_label==y_test2')/length(y_test2);
        svmModel = model;
%     end    
    % accuracy
    fprintf('\naccuracy is %f\n',mean(accuracy));
    accuracyMat(ci) = mean(accuracy);
end
rmpath(genpath('../3rdParty/liblinear-1.93/matlab'));

% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));
% ly = unique(y2_train);
% svmModel = cell(1,length(ly));
% for i=1:length(ly)
%     y_train2 = y2_train;
%     y_train2(y2_train~=ly(i))=0;
%     y_test2 = y2_test;
%     y_test2(y2_test~=ly(i))=0;
%     model = svmtrain(y_train2',X2_train',sprintf('-s 0 -t 0 -c %d',C));
%     [predict_label, accuracy, prob_estimates] = svmpredict(y_test2', X2_test', model);
%     accuracy(1)
%     svmModel{i} = model;
% end
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));

% % scale data
% addpath(genpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-3.17'));
% libsvmwrite('feat_train',y2_train',sparse(X2_train'));
% libsvmwrite('feat_test',y2_test',sparse(X2_test'));
% libsvmwrite('feat_validate',y2_validate',sparse(X2_validate'));
% system('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17/svm-scale -l 0 -u 1 -s range feat_train > feat_train_scale');
% system('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17/svm-scale -l 0 -u 1 -r range feat_test > feat_test_scale');
% system('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17/svm-scale -l 0 -u 1 -r range feat_validate > feat_validate_scale');
% [y2_train_scale, X2_train_scale] = libsvmread('feat_train_scale');
% [y2_test_scale, X2_test_scale] = libsvmread('feat_test_scale');
% [y2_validate_scale, X2_validate_scale] = libsvmread('feat_validate_scale');
% rmpath(genpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-3.17'));
% X2_train = full(X2_train_scale)';
% y2_train = y2_train_scale';
% X2_test = full(X2_test_scale)';
% y2_test = y2_test_scale';
% X2_validate = full(X2_validate_scale)';
% y2_validate = y2_validate_scale';

%% chi square svm
% addpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
% % use cross validation to decide parameters
% GInd = -7:1;
% CInd = -3:5;
% G = 5.^GInd;
% C = 5.^CInd;
% % G = 1;
% % C = 1e2;
% 
% % N = size(hFeat_train,2);
% % K = 5;
% % ind = crossvalind('Kfold',N,K);
% % accuracyMat = zeros(length(G),length(C),K);
% 
% accuracyMat = zeros(length(G),length(C));
% % for fi = 1:K
%     
% %     X2_train = hFeat_train(:,ind~=fi);
% %     y2_train = al_train(ind~=fi)';
% %     X2_validate = hFeat_train(:,ind==fi);
% %     y2_validate = al_train(ind==fi)';
% %     X2_test = hFeat_test;
% %     y2_test = al_test';
%     
%     for gi = 1:length(G)
%         for ci = 1:length(C)
%             model = svmtrain_chi2(y2_train',X2_train',sprintf('-t 5 -g %f -c %d -q',G(gi),C(ci)));
%             [predict_label, accuracy, prob_estimates] = svmpredict_chi2(y2_test', X2_test', model);
%             accuracyMat(gi,ci) = nnz(predict_label==y2_test')/length(y2_test);
% 
%             % accuracy
%             fprintf('\naccuracy is %f\n',accuracy(1));
%         end
%     end
%     
% % end
% % meanAccuracyMat = mean(accuracyMat,3);
% % meanAccuracyMat
% % [sub_a,sub_b] = find(meanAccuracyMat==max(max(meanAccuracyMat)));
% [sub_a,sub_b] = find(accuracyMat==max(max(accuracyMat)));
% 
% % G = 1;
% % C = 25;
% model = svmtrain_chi2(y2_train',X2_train',sprintf('-t 5 -g %f -c %d -q',G(sub_a(1)),C(sub_b(1))));
% [predict_label, ~, prob_estimates] = svmpredict_chi2(y2_test', X2_test', model);
% accuracy = nnz(predict_label==y2_test')/length(y2_test)

% save('accuracy_action01_06_person01_26_scene01_01_20131118','accuracy','y2_test','predict_label');

% label_gt = [label_gt; y2_test];
% label_pred = [label_pred; predict_label];
% end
%
% disp('accuracy is ');
% nnz(label_gt-label_pred==0)/length(label_gt)

% rmpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
