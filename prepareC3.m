function [trainC2,testC2,trainLabels,testLabels,trainFiles,testFiles] = ...
  prepareC3(c2Dir,patchSet,trainingFactor,minPerClass)
% function [trainC2,testC2,trainLabels,testLabels,trainFiles,testFiles] = ...
%   prepareC3(c2Dir,patchSet,trainingFactor,minPerClass)
    allCaches = dir([c2Dir '*' patchSet '.c2.mat']);
    allCacheFiles = strcat(c2Dir,{allCaches.name}');
    shuffledClasses = allCacheFiles(randperm(length(allCacheFiles)));
    goodClasses = checkForMinExamples(shuffledClasses,minPerClass);
    cutoff = min(length(goodClasses), ...
                 floor(length(goodClasses)*trainingFactor));
    trainFiles = shuffledClasses(goodClasses(1:cutoff));
    testFiles = shuffledClasses(goodClasses(cutoff+1:end));
    [trainC2,trainLabels] = prepareC3Helper(trainFiles);
    [testC2,testLabels] = prepareC3Helper(testFiles);
end


function [goodClasses,nImgs] = checkForMinExamples(cacheFiles,minPerClass)
    nClasses = length(cacheFiles);
    nImgs = zeros(nClasses,1);
    for iClass = 1:nClasses
        load(cacheFiles{iClass},'c2');
        nImgs(iClass) = size(c2,2);
    end
    goodClasses = find(nImgs >= minPerClass);
    nImgs = nImgs(goodClasses);
end


function [allC2,allLabels] = prepareC3Helper(cacheFiles)
    nClasses = length(cacheFiles);
    nImgs = zeros(nClasses,1);
    c2s = cell(nClasses,1);
    for iClass = 1:nClasses
        load(cacheFiles{iClass},'c2');
        nImgs(iClass) = size(c2,2);
        c2s{iClass} = c2;
        clear c2;
    end
    allLabels = zeros(nClasses,sum(nImgs));
    for iClass = 1:nClasses
        start = sum(nImgs(1:iClass-1))+1;
        stop = start+nImgs(iClass)-1;
        allLabels(iClass,start:stop) = 1;
    end
    allC2 = [c2s{:}];
end
