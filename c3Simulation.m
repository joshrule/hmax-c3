function c3Simulation(outDir)
% c3Simulation(outDir)
    ensureDir(outDir);

    if ~exist([outDir 'params.mat'],'file')
        params = getParams();  
        save([outDir 'params.mat'],'params','-v7.3');
    else
        load([outDir 'params.mat']);
    end
    p = params;
    fprintf('params generated\n');

    if ~exist([outDir 'c2.mat'],'file') || ~exist([outDir 'splits.mat'],'file')
        [trainC2,testC2,trainLabels,testLabels,trainFiles,testFiles] = ...
        prepareC3(p.c2Dir,p.patchSet,p.trainingFactor,p.minPerClass);
        cvsplit = cv(testLabels,p.nTrainingExamples,p.nRuns);
        save([outDir 'c2.mat'],'trainC2','testC2','-v7.3');
        save([outDir 'splits.mat'],'trainLabels','testLabels', ...
             'trainFiles','testFiles','cvsplit','-v7.3');
    else
        load([outDir 'c2.mat']);
        load([outDir 'splits.mat']);
    end
    fprintf('test/train splits generated\n');

    if ~exist([outDir 'models.mat'],'file')
        models = trainC3(trainC2,trainLabels,p.method,p.options);
        save([outDir 'models.mat'],'models','-v7.3');
    else
        load([outDir 'models.mat']);
    end
    fprintf('models generated\n');

    if ~exist([outDir 'c3.mat'],'file')
        c3 = testC3(testC2,models,p.method,trainLabels(1));
        save([outDir 'c3.mat'],'c3','-v7.3');
    else
        load([outDir 'c3.mat']);
    end
    fprintf('C3 generated\n');

    if ~exist([outDir 'c2-evaluation.mat'],'file')
        [aucsC2,dprimesC2] = evaluatePerformance(testC2,testLabels,cvsplit, ...
                                                 p.method,p.options, ...
                                                 size(testC2,1),[]);
        save([outDir 'c2-evaluation.mat'],'aucsC2','dprimesC2','-v7.3');
    end
    fprintf('C2 performance evaluations generated\n');

    if ~exist([outDir 'c3-evaluation.mat'],'file')
        [aucsC3,dprimesC3,modelsC3] = evaluatePerformance( ...
          c3,testLabels,cvsplit,p.method,p.options,size(c3,1),[]);
        save([outDir 'c3-evaluation.mat'],'aucsC3','dprimesC3','modelsC3', ...
          '-v7.3');
    end
    fprintf('C3 performance evaluations generated\n');

%   if ~exist([outDir 'c2c3-evaluation.mat'],'file')
%       [aucsC2C3,dprimesC2C3,modelsC2C3] = evaluatePerformance( ...
%         [testC2; c3],testLabels,cvsplit,p.method,p.options, ...
%         size(c3,1)+size(testC2,1),[]);
%       save([outDir 'c2c3-evaluation.mat'],'aucsC2C3','dprimesC2C3', ...
%         'modelsC2C3','-v7.3');
%   end
%   fprintf('C2+C3 performance evaluations generated\n');
end

function params = getParams()
    params.c2Dir = '/home/joshrule/maxlab/image-sets/image-net/c2CacheClean/';
    params.patchSet = 'universalPatches400PerSize';
    params.trainingFactor = 940/990;
    params.minPerClass = 150;
    params.options = '-s 0 -t 0 -c 0.1 -b 1 -q';
    params.nTrainingExamples = [2 4 8 16 32 64 128 160 192 224];
    params.nRuns = 10;
    params.method = 'svm';
end
