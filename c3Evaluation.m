function c3Evaluation(outDir,params)
% c3Evaluation(outDir,params)
%
% Perform a series of performance evaluations on the C2 and C3 activations
% generated during a full C3 simulation. All data is written to files in
% 'outDir'.
%
% outDir: string, directory to which to write the evaluation information
% params: struct, the parameters governing the simulation itself, see 
%   'c3Simulation.m' for details
    p = params;

    load([outDir 'c2.mat'],testC2);
    load([outDir 'splits.mat'],'cvsplit','testLabels');

    if ~exist([outDir 'c2-evaluation.mat'],'file')
        [aucsC2,dprimesC2] = evaluatePerformance(testC2,testLabels,cvsplit, ...
                                                 p.method,p.testOptions, ...
                                                 size(testC2,1),[]);
        save([outDir 'c2-evaluation.mat'],'aucsC2','dprimesC2','-v7.3');
    end
    fprintf('C2 performance evaluations generated\n');

    if ~exist([outDir 'c3-evaluation.mat'],'file')
        [aucsC3,dprimesC3,modelsC3] = evaluatePerformance( ...
          c3,testLabels,cvsplit,p.method,p.testOptions,size(c3,1),[]);
        save([outDir 'c3-evaluation.mat'],'aucsC3','dprimesC3','modelsC3', ...
          '-v7.3');
    end
    fprintf('C3 performance evaluations generated\n');

    if ~exist([outDir 'c2c3-evaluation.mat'],'file')
        [aucsC2C3,dprimesC2C3,modelsC2C3] = evaluatePerformance( ...
          [testC2; c3],testLabels,cvsplit,p.method,p.testOptions, ...
          size(c3,1)+size(testC2,1),[]);
        save([outDir 'c2c3-evaluation.mat'],'aucsC2C3','dprimesC2C3', ...
          'modelsC2C3','-v7.3');
    end
    fprintf('C2+C3 performance evaluations generated\n');
end
