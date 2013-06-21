function [trainC2,testC2,trainLabels,testLabels,trainFiles,testFiles] = ...
  prepareC3(trainDir,testDir,patchSet,trainN,testN,minPerClass)
% [trainC2,testC2,trainLabels,testLabels,trainFiles,testFiles] = ...
%   prepareC3(trainDir,testDir,patchSet,trainN,testN,minPerClass)
%
% Create the testing/training split data for a C3 simulation
%
% trainDir: string, location of training class c2 caches
% testDir: string, location of test class c2 caches
% patchSet: string, what patch set should the classes have been cached with?
% trainN: scalar, number of training categories (i.e. vocabulary concepts)
% testN: scalar, number of test categories
% minPerClass: scalar, minimum number of examples allowed in each training and
%   testing class
%
% trainC2: nFeatures x nTrainingImgs array, C2 responses to train C3 classifiers
% testC2: nFeatures x nTestingImgs array, C2 responses to test C3 classifiers
% trainLabels: trainN x nTrainingImgs array, class labels for training images
% testLabels: testN x nTestingImgs array, class labels for testing images
% trainFiles: cell array of strings, the c2 cache files used to make 'trainC2'
% testFiles: cell array of strings, the c2 cache files used to make 'testC2'
    [trainFiles,trainC2,trainLabels] = ...
       prepareC3Helper(trainDir,patchSet,minPerClass,trainN);
    [testFiles,testC2,testLabels] = ...
      prepareC3Helper(testDir,patchSet,minPerClass,testN);
end

function [goodClasses,nImgs] = checkForMinExamples(cacheFiles,minPerClass)
% [goodClasses,nImgs] = checkForMinExamples(cacheFiles,minPerClass)
%
% check a cacheFile to ensure it has enough examples to be useful
%
% cacheFiles: cell array of strings, a list of c2 cache files
% minPerClass: scalar, the minimum number of images required to be useful
%
% goodClasses: cell array of strings, the cache files with minPerClass images
% nImgs: the actual number of images in each cache file listed in 'goodClasses'
    nClasses = length(cacheFiles);
    nImgs = zeros(nClasses,1);
    for iClass = 1:nClasses
        load(cacheFiles{iClass},'c2');
        nImgs(iClass) = size(c2,2);
    end
    goodClasses = find(nImgs >= minPerClass);
    nImgs = nImgs(goodClasses);
end

function [nFiles,nC2,nLabels] = prepareC3Helper(cacheDir,patchSet,minImgs,N)
% [nFiles,nC2,nLabels] = prepareC3Helper(cacheDir,patchSet,minImgs,N)
%
% the actual work of preparing the splits
%
% cacheDir: string, the directory holding relevant cacheFiles
% patchSet: string, only use caches created with this patch set
% minImgs: scalar, the minimum number of images required to be useful
% N: scalar, number of cacheFiles/categories to use
%
% nFiles: cell array of strings, the cache files used
% nC2: nFeatures x nImgs array, C2 matrix made by concatenating the cache files
% nLabels: N x nImgs array, class labelings for each image
    caches = dir([cacheDir '*' patchSet '.c2.mat']);
    cacheFiles = strcat(cacheDir,{caches.name}');
    shuffledClasses = cacheFiles(randperm(length(cacheFiles)));
    goodClasses = checkForMinExamples(shuffledClasses,minImgs);
    nFiles = shuffledClasses(goodClasses(1:(min(length(goodClasses),N))));

    nClasses = length(nFiles);
    nImgs = zeros(nClasses,1);
    c2s = cell(nClasses,1);
    for iClass = 1:nClasses
        load(nFiles{iClass},'c2');
        nImgs(iClass) = size(c2,2);
        c2s{iClass} = c2;
        clear c2;
    end
    nLabels = zeros(nClasses,sum(nImgs));
    for iClass = 1:nClasses
        start = sum(nImgs(1:iClass-1))+1;
        stop = start+nImgs(iClass)-1;
        nLabels(iClass,start:stop) = 1;
    end
    nC2 = [c2s{:}];
end
