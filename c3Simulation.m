function c3Simulation(outDir,params)
% c3Simulation(outDir)
    ensureDir(outDir);

    if ~exist([outDir 'params.mat'],'file')
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
        models = trainC3(trainC2,trainLabels,p.method,p.trainOptions);
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
end
