clear;clc;
addpath([ 'C:\Users\Aven\Desktop\Aven\Random-Forest-Matlab-master\Random-Forest-Matlab-master']);
addpath(['C:\Users\Aven\Desktop\Aven\MATLAB_scripts_functions\MATLAB_scripts_functions'])
fea_video = xlsread('video');
train = xlsread('train_split_Depression_AVEC2017.csv');
dev = xlsread('dev_split_Depression_AVEC2017.csv');
test = xlsread('test_split_Depression_AVEC2017.csv');
train_list =train(:,[1]);
dev_list =dev(:,[1]);
test_list =test(:,[1]);
video_train = fea_video(train_list ,:);
video_test = fea_video(test_list ,:);
video_dev = fea_video(dev_list ,:);
y_train = train(:,2); y_trainScore = train(:,3); 
y_dev = dev(:,2);  y_devScore = dev(:,3);
opts= struct;
opts.depth= 9;
opts.numTrees= 100;
opts.numSplits= 5;
opts.verbose= true;
opts.classifierID= 2; % weak learners to use. Can be an array for mix of weak learners too
model = forestTrain(video_train, y_train, opts);
yhat = forestTest(model, video_dev);

fprintf('Classifier distributions:\n');
classifierDist= zeros(1, 4);
unused= 0;
for i=1:length(model.treeModels)
    for j=1:length(model.treeModels{i}.weakModels)
        cc= model.treeModels{i}.weakModels{j}.classifierID;
        if cc>1 %otherwise no classifier was used at that node
            classifierDist(cc)= classifierDist(cc) + 1;
        else
            unused= unused+1;
        end
    end
end
fprintf('%d nodes were empty and had no classifier.\n', unused);
for i=1:4
    fprintf('Classifier with id=%d was used at %d nodes.\n', i, classifierDist(i));
end
[yhat, ysoft] = forestTest(model, video_dev);
sprintf('Test accuracy: %f\n', mean(yhatTrain==y_dev));