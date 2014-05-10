function c3Simulation(outDir,params)
% c3Simulation(outDir,params)
%
% Given a set of parameters, load up a set of C2 responses for testing and
% training, use the training matrix to generate a set of C3 classifiers, and use
% those C3 classifiers to generate C3 activations for the C2 test data. Save all
% data to disk.
%
% outDir: string, directory to which to write the simulation data
% params: struct, the parameters governing the simulation itself with the
% following structure:
%   home: top-level directory holding code, data, etc.
%   trainDir: string, location of training class c2 caches
%   trainN: scalar, number of training categories (i.e. vocabulary concepts)
%   testDir: string, location of test class c2 caches
%   testN: scalar, number of test categories
%   patchSet: string, what patch set should the classes have been cached with?
%   minPerClass: scalar, minimum number of examples allowed in each training and
%     testing class
%   trainOptions: struct with the following fields, for training the c3 units
%       svmTrainFlags: options for training C3 classifiers with hard mining
%       svmTestFlags: options for testing C3 classifiers with hard mining
%       alpha: scalar, governs the growth of the hard negative mining
%       startPerIter: scalar, number of images in the first mining iteration
%       threshold: scalar, probability above which a negative is "hard"
%   testOptions: string or double, options for the test classifier
%   nTrainingExamples: double array, total numbers of training examples to use
%     in evaluation (e.g. [16 32 64 128])
%   nRuns: scalar, the number of cross-validation runs
%   method: string, 'gb' or 'svm', the classifier to use

    rng(0,'twister'); 
    fprintf('Pseudorandom Number Generator Reset\nrng(0,''twister'')\n\n');
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
          prepareC3(p.trainDir,p.testDir,p.patchSet,p.trainN,p.testN, ...
	      p.minPerClass);
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
    	% some randomness involved...
        models = trainC3(trainC2,trainLabels,p.method,p.trainOptions);
        save([outDir 'models.mat'],'models','-v7.3');
    else
        load([outDir 'models.mat']);
    end
    fprintf('models generated\n');

    if ~exist([outDir 'c3.mat'],'file')
        c3 = testC3(testC2,models,p.method);
        save([outDir 'c3.mat'],'c3','-v7.3');
    else
        load([outDir 'c3.mat']);
    end
    fprintf('C3 generated\n');
end
