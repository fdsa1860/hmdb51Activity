% test hankel and kmeans
% Xikang Zhang, 04/01/2014

% function testHankelFeat
clear;clc;close all;

params.num_km_init_word = 3;
params.MaxInteration = 3;
params.labelBatchSize = 300000;
params.actualFilterThreshold = -1;
params.find_labels_mode = 'DF';
nCenter = 300;
% ncenter = 100;

addpath(genpath('../3rdParty/hankelet-master/hankelet-master'));
addpath(genpath(getProjectBaseFolder));

% % % Load data
% file = 'CodePool_2014_0401.mat';
% load(fullfile('../expData',file));
% 
% % subtract mean
% xm = mean(X(1:2:end,:));
% ym = mean(X(2:2:end,:));
% Xm = kron(ones(size(X,1)/2,1),[xm;ym]);
% X = X - Xm;
% 
% % kmeans to learn cluster centers
% trainCenter = cell(1, params.num_km_init_word);
% for i = 1 : params.num_km_init_word
%     
%     assert(size(X,1)==30);
%     [~, trainCenter{i} trainClusterMu trainClusterSigma trainClusterNum] = litekmeans_subspace(X, ncenter,params);
%     
%     params.trainClusterInfo{i}.mu = trainClusterMu;
%     params.trainClusterInfo{i}.sigma = trainClusterSigma;
%     params.trainClusterInfo{i}.num = trainClusterNum;
%     
%     params.trainClusterNum{i} = size(trainCenter{i}, 2);
%     
% end
% 
% % labeling
% params = cal_cluster_info(params);

% % save kmeansWords300_action01_06_person01_26_scene01_04_20131118t params trainCenter;
% load ../expData/kmeansWords300_hmdb51_20140401;

% % get hankelet features
% load('../expData/hmdb51_fileSplit1_20140326.mat','trainSet', ...
%     'testSet', 'trainLabel', 'testLabel');
% 
% hFeat_train = zeros(nCenter,length(trainSet));
% al_train = trainLabel;
% indToErase = [];
% for i=1:length(trainSet)
%     traj = load(trainSet{i});
%     if isempty(traj)
%         indToErase = [indToErase i];
%         continue;
%     end
%     X = traj(:,2:end)';
%     [label1, dis, class_hist] = find_weight_labels_df_HHp_newProtocal({trainCenter{3}},X, params);
%     hFeat_train(:,i) = class_hist';
%     fprintf('%d of %d training files are processed.\n',i,length(trainSet));
% end
% hFeat_train(:,indToErase) = [];
% al_train(indToErase) = [];
% 
% hFeat_test = zeros(nCenter,length(testSet));
% al_test = testLabel;
% indToErase = [];
% for i=1:length(testSet)
%     traj = load(testSet{i});
%     if isempty(traj)
%         indToErase = [indToErase i];
%         continue;
%     end
%     X = traj(:,2:end)';
%     [label1, dis, class_hist] = find_weight_labels_df_HHp_newProtocal({trainCenter{3}},X, params);
%     hFeat_test(:,i) = class_hist';
%     fprintf('%d of %d testing files are processed.\n',i,length(testSet));
% end
% hFeat_test(:,indToErase) = [];
% al_test(indToErase) = [];

% %     save hFeat300_hmdb51_20140402 hFeat_train al_train hFeat_test al_test
load ../expData/hFeat300_hmdb51_20140402



% normalization
sum_hFeat_train = sum(hFeat_train,1);
hFeat_train = hFeat_train./bsxfun(@times,sum_hFeat_train,ones(size(hFeat_train,1),1));
sum_hFeat_test = sum(hFeat_test,1);
hFeat_test = hFeat_test./bsxfun(@times,sum_hFeat_test,ones(size(hFeat_test,1),1));

% %% train a SVM problem
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

% %% train a SVM problem using one versus all, liblinear
% Cind = -1:10;
% C = 2.^Cind;
% % C = 512;
% accuracyMat = zeros(length(G),length(C));
% for ci = 1:length(C)
%     addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/liblinear-1.93/matlab'));
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
%         model = train(y_train2',sparse(X2_train'),sprintf('-s 2 -c %d',C(ci)));
% %         [predict_label, ~, prob_estimates] = predict(y_validate2', sparse(X2_validate'), model);
% %         accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
%             [predict_label, ~, prob_estimates] = predict(y_test2', sparse(X2_test'), model);
%             accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
%         svmModel{i} = model;
%     end
%     rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/liblinear-1.93/matlab'));
%     % accuracy
%     fprintf('\naccuracy is %f\n',mean(accuracy));
%     accuracyMat(ci) = mean(accuracy);
% end

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

N = size(hFeat_train,2);
% K = 5;
% ind = crossvalind('Kfold',N,K);

K = 1;

accuracyMat = zeros(length(G),length(C),K);
for fi = 1:K
    
%     X2_train = hFeat_train(:,ind~=fi);
%     y2_train = al_train(ind~=fi)';
%     X2_validate = hFeat_train(:,ind==fi);
%     y2_validate = al_train(ind==fi)';
%     X2_test = hFeat_test;
%     y2_test = al_test';
    
    for gi = 1:length(G)
        for ci = 1:length(C)
%             model = svmtrain_chi2(y2_train',X2_train',sprintf('-t 5 -g %f -c %d -q',G(gi),C(ci)));
%             [predict_label, accuracy, prob_estimates] = svmpredict_chi2(y2_validate', X2_validate', model);
%             accuracyMat(gi,ci,fi) = nnz(predict_label==y2_validate')/length(y2_validate);
            model = svmtrain_chi2(al_train,hFeat_train',sprintf('-t 5 -g %f -c %d -q',G(gi),C(ci)));
            [predict_label, accuracy, prob_estimates] = svmpredict_chi2(al_test, hFeat_test', model);
            accuracyMat(gi,ci,fi) = nnz(predict_label==al_test)/length(al_test);

            % accuracy
            fprintf('\naccuracy is %f\n',accuracy(1));
        end
    end
    
end
meanAccuracyMat = mean(accuracyMat,3);
meanAccuracyMat
[sub_a,sub_b] = find(meanAccuracyMat==max(max(meanAccuracyMat)));

% G = 1e-4;
% C = 100;
% G = 1;
% C = 5;
model = svmtrain_chi2(al_train,hFeat_train',sprintf('-t 5 -g %f -c %d -q',G(sub_a),C(sub_b)));
[predict_label, ~, prob_estimates] = svmpredict_chi2(al_test, hFeat_test', model);
accuracy = nnz(predict_label==y2_test')/length(y2_test)

% save('accuracy_action01_06_person01_26_scene01_01_20131118','accuracy','y2_test','predict_label');

% label_gt = [label_gt; y2_test];
% label_pred = [label_pred; predict_label];
% end
%
% disp('accuracy is ');
% nnz(label_gt-label_pred==0)/length(label_gt)

rmpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
