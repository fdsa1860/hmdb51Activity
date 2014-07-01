% test brief feature
% Xikang Zhang, 03/19/2014

% function testBriefFeat
clear;clc;close all;

% load('../expData/hmdb51_tracklet_fileSplit2.mat','trainSet', ...
%     'testSet', 'trainLabel', 'testLabel');
% D = 30;
% 
% % get brief features
% bSize = 8;
% accuracyWrtBitSize = zeros(length(bSize),1);
% % for m=1:length(bSize)
% m = 1;
% rng(0);
% while(true)
%     patt = randi(D,bSize(m),2);
%     if all(patt(:,2)-patt(:,1))
%         break;
%     end
% end
% while(true)
%     patt1 = randi(D-2,bSize(m),2);
%     if all(patt1(:,2)-patt1(:,1))
%         break;
%     end
% end
% while(true)
%     patt2 = randi(D-4,bSize(m),2);
%     if all(patt2(:,2)-patt2(:,1))
%         break;
%     end
% end
% 
% diff1 = zeros(30,28);
% diff1(1:28,1:28) = -eye(28);
% diff1(3:30,1:28) = diff1(3:30,1:28) + eye(28);
% diff2 = diff1(1:end-2,1:end-2);
% 
% bFeatSize = 2^bSize;
% 
% bFeat = zeros(bFeatSize,length(trainSet));
% al = trainLabel;
% indToErase = [];
% for i=1:length(trainSet)
%     traj = load(trainSet{i});
%     if isempty(traj)
%         indToErase = [indToErase i];
%         continue;
%     end
%     X = traj(:,2:end)';
%     class_hist = findBriefHist(X,patt,true);
%     bFeat(:,i) = class_hist';
%     fprintf('%d of %d training files are processed.\n',i,length(trainSet));
% end
% bFeat(:,indToErase) = [];
% al(indToErase) = [];
% bFeat_train = bFeat;
% al_train = al;
% 
% bFeat = zeros(bFeatSize,length(testSet));
% al = testLabel;
% indToErase = [];
% for i=1:length(testSet)
%     traj = load(testSet{i});
%     if isempty(traj)
%         indToErase = [indToErase i];
%         continue;
%     end
%     X = traj(:,2:end)';
%     class_hist = findBriefHist(X,patt,true);
%     bFeat(:,i) = class_hist';
%     fprintf('%d of %d testing files are processed.\n',i,length(testSet));
% end
% bFeat(:,indToErase) = [];
% al(indToErase) = [];
% bFeat_test = bFeat;
% al_test = al;
% %    save bFeat_20140326 bFeat_train bFeat_test al_train al_test;
load ../expData/bFeat_split2_20140407;

X2_train = bFeat_train;
X2_test = bFeat_test;
y2_train = al_train;
y2_test = al_test;

% % train a SVM problem
% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));
% model = svmtrain(y2_train',X2_train_normalized',sprintf('-s 0 -c %d',C));
% [predict_label, accuracy, prob_estimates] = svmpredict(y2_test', X2_test_normalized', model);
% y2_predict = predict_label';
% accuracy(1)
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));

% % train a SVM problem
% addpath(genpath('../3rdParty/liblinear-1.93/matlab'));
% model = train(y2_train',sparse(X2_train'),sprintf('-s 2 -c %d',C));
% [predict_label, accuracy, prob_estimates] = predict(y2_test', sparse(X2_test'), model);
% y2_predict = predict_label';
% accuracy(1)
% rmpath(genpath('../3rdParty/liblinear-1.93/matlab'));

% % train a SVM problem using one versus all
% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/liblinear-1.93/matlab'));
% ly = unique(y2_train);
% svmModel = cell(1,length(ly));
% for i=1:length(ly)
%     y_train2 = y2_train;
%     y_train2(y2_train~=ly(i))=0;
%     y_test2 = y2_test;
%     y_test2(y2_test~=ly(i))=0;
%     model = train(y_train2',sparse(X2_train'),sprintf('-s 2 -c %d',C));
%     [predict_label, accuracy, prob_estimates] = predict(y_test2', sparse(X2_test'), model);
%     accuracy(1)
%     svmModel{i} = model;
% end
% % save svmModel1024 svmModel
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/liblinear-1.93/matlab'));

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
addpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
% use cross validation to decide parameters
GInd = -7:1;
CInd = -3:5;
G = 5.^GInd;
C = 5.^CInd;
% G = 1;
% C = 1e2;

N = size(bFeat_train,2);
K = 5;
ind = crossvalind('Kfold',N,K);

accuracyMat = zeros(length(G),length(C),K);
for fi = 1:K
    
    X2_train = bFeat_train(:,ind~=fi);
    y2_train = al_train(ind~=fi)';
    X2_validate = bFeat_train(:,ind==fi);
    y2_validate = al_train(ind==fi)';
    X2_test = bFeat_test;
    y2_test = al_test';
    
    for gi = 1:length(G)
        for ci = 1:length(C)
            model = svmtrain_chi2(y2_train',X2_train',sprintf('-t 5 -g %f -c %d -q',G(gi),C(ci)));
            [predict_label, accuracy, prob_estimates] = svmpredict_chi2(y2_validate', X2_validate', model);
            accuracyMat(gi,ci,fi) = nnz(predict_label==y2_validate')/length(y2_validate);

            % accuracy
            fprintf('\naccuracy is %f\n',accuracy(1));
        end
    end
    
end
meanAccuracyMat = mean(accuracyMat,3);
meanAccuracyMat
[sub_a,sub_b] = find(meanAccuracyMat==max(max(meanAccuracyMat)));

model = svmtrain_chi2(y2_train',X2_train',sprintf('-t 5 -g %f -c %d -q',G(sub_a(1)),C(sub_b(1))));
[predict_label, ~, prob_estimates] = svmpredict_chi2(y2_test', X2_test', model);
accuracy = nnz(predict_label==y2_test')/length(y2_test)

% save('accuracy_action01_06_person01_26_scene01_01_20131118','accuracy','y2_test','predict_label');

% label_gt = [label_gt; y2_test];
% label_pred = [label_pred; predict_label];
% end
%
% disp('accuracy is ');
% nnz(label_gt-label_pred==0)/length(label_gt)

rmpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');