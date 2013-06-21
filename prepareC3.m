function [trainC2,testC2,trainLabels,testLabels,trainFiles,testFiles] = ...
  prepareC3(trainDir,testDir,patchSet,trainN,testN,minPerClass)
% [trainC2,testC2,trainLabels,testLabels,trainFiles,testFiles] = ...
% prepareC3(trainDir,testDir,patchSet,trainN,testN,minPerClass)
    [trainFiles,trainC2,trainLabels] = ...
       prepareC3Helper(trainDir,patchSet,minPerClass,trainN);
    [testFiles,testC2,testLabels] = ...
      prepareC3Helper(testDir,patchSet,minPerClass,testN);
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

function [nFiles,nC2,nLabels] = prepareC3Helper(cacheDir,patchSet,minImgs,N)
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
