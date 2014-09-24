% test fisher vector
% Xikang Zhang, 06/04/2014

% function testFisherFeat
clear;clc;close all;
addpath('../3rdParty/yael_matlab_mac64_v401');

% nInitCenter = 256;
nInitCenter = 512;

%% Load data
file = 'CodePool_20140417.mat'; % 300000
load(fullfile('../expData',file));
X = X(40:135,:);      % HOG
% X = X(136:243,:);   % HOF
% X = X(244:339,:);       % MBHx
% X = X(340:435,:);       % MBHy
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
% [Score, pcaEigVec, pcaEigVal, pcaCenter] = yael_pca (X,D2);
% toc
% save pca256_hmdb51_MBHy_20140909 pcaEigVec pcaEigVal pcaCenter;
% load ../expData/pca256_hmdb51_hog_20140903;

% X2 = diag(pcaEigVal.^-0.5)*pcaEigVec'*bsxfun(@minus,X,pcaCenter);
% % X2 = pcaEigVec'*bsxfun(@minus,X,pcaCenter);
% % X2 = X;
% tic
% [means, covariances, priors] = vl_gmm(X2, nInitCenter);
% toc
% % tic;
% % [priors, means, covariances] = yael_gmm(single(X2), nInitCenter);
% % toc
% % save gmm512_hmdb51_hog_PCA_Whiten_20140922 means covariances priors;
% load ../expData/gmm512_hmdb51_hog_PCA_Whiten_20140922;

%% get dense trajectory BOW features
% load('../expData/hmdb51_trackletOrig_fileSplit1.mat','allDataSet', ...
%     'allDataLabel','allDataSplit');
% startTime = tic;
% 
% D2 = size(means,1);
% fFeat = zeros(2*nInitCenter*D2,length(allDataSet));
% % fFeat = zeros(nInitCenter*D2,length(allDataSet));
% % load fFeat3044;
% indToErase = false(size(allDataLabel));
% tic
% parfor i=1:length(allDataSet)
% % for i=3045:length(allDataSet)
%     
%     traj = load(allDataSet{i});
%     
%     if isempty(traj)
%         indToErase(i) = true;
%         continue;
%     end
%     
%     X = traj(:,41:41+96-1)';    % HOG, ignore the first 10 elements
% %     X = traj(:,137:137+108-1)';    % HOF, ignore the first 10 elements
% %     X = traj(:,245:245+96-1)';  % MBHx
% %     X = traj(:,341:341+96-1)';  % MBHy
%     % pca projection
%     X2 = diag(pcaEigVal.^-0.5)*pcaEigVec'*bsxfun(@minus,X,pcaCenter);
% %     X2 = pcaEigVec'*bsxfun(@minus,X,pcaCenter);
% %     X2 = X;
%     % get fisher vector
%     class_hist = vl_fisher(X2, means, covariances, priors);
% %     class_hist = yael_fisher(single(X2), priors, means, covariances,'sigma');
% %     class_hist = yael_fisher(single(X2), priors, means, covariances);
%     fFeat(:,i) = class_hist;
%     
% %     save fFeat fFeat
%     fprintf('%d files are processed.\n',i);
% 
% end
% toc
% 
% fFeat(:,indToErase) = [];
% allDataLabel(indToErase) = [];
% allDataSplit(indToErase) = [];
% 
% toc(startTime)
% 
% save('fFeat512_hmdb51_hog_PCA_Whiten_vlfeat_fisher_20140922','fFeat','allDataLabel','allDataSplit','-v7.3');
% load ../expData/fFeat512_hmdb51_hog_PCA_Whiten_vlfeat_fisher_20140922.mat

%% load all features
% load ../expData/fFeat256_hmdb51_hog_PCA_Whiten_vlfeat_fisher_20140903.mat
load ../expData/fFeat512_hmdb51_hog_PCA_Whiten_vlfeat_fisher_20140922.mat
fFeatHOG = fFeat;
labelHOG = allDataLabel;
fFeatHOG = powerNormalization(fFeatHOG);
fFeatHOG = l2Normalization(fFeatHOG);
% load ../expData/fFeat256_hmdb51_L108_20140826.mat
load ../expData/fFeat512_hmdb51_L108_PCA_Whiten_vlfeat_fisher_20140919.mat
fFeatHOF = fFeat;
labelHOF = allDataLabel;
fFeatHOF = powerNormalization(fFeatHOF);
fFeatHOF = l2Normalization(fFeatHOF);
% load ../expData/fFeat256_hmdb51_MBHx_PCA_Whiten_vlfeat_fisher_20140908.mat
load ../expData/fFeat512_hmdb51_MBHx_PCA_Whiten_vlfeat_fisher_20140915.mat
fFeatMBHx = fFeat;
labelMBHx = allDataLabel;
fFeatMBHx = powerNormalization(fFeatMBHx);
fFeatMBHx = l2Normalization(fFeatMBHx);
% load ../expData/fFeat256_hmdb51_MBHy_PCA_Whiten_vlfeat_fisher_20140909.mat
load ../expData/fFeat512_hmdb51_MBHy_PCA_Whiten_vlfeat_fisher_20140910.mat
fFeatMBHy = fFeat;
labelMBHy = allDataLabel;
fFeatMBHy = powerNormalization(fFeatMBHy);
fFeatMBHy = l2Normalization(fFeatMBHy);

clear fFeat allDataLabel;

fFeat = [fFeatHOG;fFeatHOF;fFeatMBHx;fFeatMBHy];
allDataLabel = [labelHOG;labelHOF;labelMBHx;labelMBHy];

%% split data into train and test data
X2_train = fFeat(:,allDataSplit==1);
y2_train = allDataLabel(allDataSplit==1)';
X2_test = fFeat(:,allDataSplit==2);
y2_test = allDataLabel(allDataSplit==2)';


%% normalization

% % intra normalization
% intra_X2_train = sum(abs(X2_train),2).^0.5;
% X2_train = X2_train./bsxfun(@times,intra_X2_train,ones(1,size(X2_train,2)));
% power normalization
X2_train = sign(X2_train).*(abs(X2_train).^0.5);
% L2 normalization
L2_X2_train = sum(X2_train.^2,1).^0.5;
X2_train = X2_train./bsxfun(@times,L2_X2_train,ones(size(X2_train,1),1));

% % intra normalization
% X2_test = X2_test./bsxfun(@times,intra_X2_train,ones(1,size(X2_test,2)));
% power normalization
X2_test = sign(X2_test).*(abs(X2_test).^0.5);
% L2 normalization
L2_X2_test = sum(X2_test.^2,1).^0.5;
X2_test = X2_test./bsxfun(@times,L2_X2_test,ones(size(X2_test,1),1));
% fFeat = sign(fFeat).*(abs(fFeat).^0.5);
% sum_fFeat = sum(fFeat.^2,1).^0.5;
% fFeat = fFeat./bsxfun(@times,sum_fFeat,ones(size(fFeat,1),1));

%% train a SVM problem



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
addpath(genpath('../3rdParty/liblinear-1.94/matlab'));
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
rmpath(genpath('../3rdParty/liblinear-1.94/matlab'));

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
