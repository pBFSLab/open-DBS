clear; clc;
inpath = './data/DBS_2024/Parcellation152/Indi_FC/'; % Relative path
addpath(genpath('./code/From_Xiaoxuan/scripts/barplots_7nets_2bars_groupedbars')); % Relative path

sessions = {'CH1m', 'CH3m', 'CH6m', 'CH12m', 'CL1m', 'CL3m', 'CL6m', 'CL12m', 'CV1m', 'CV3m', 'CV6m', 'CV12m'};
listpath = './code/From_Xiaoxuan/scripts'; % Relative path
subjects = textread([listpath '/sub_name_num.txt'], '%s');
% subid = textread([listpath '/sub_list.txt'], '%s');
num = 152;
outpath = './data/DBS_2024/test_Retest/FC_corr/Parc152_Results/Parc_FC'; % Relative path
if ~exist(outpath, 'dir'); mkdir(outpath); end

for s = 1:length(subjects)
    subj = subjects{s};
    disp(sprintf('Loading the data for subject: %s', subj))
    
    % Define sessions based on the subject
    if s == 1
        sessions = {'PreRest', '1mRest', '1mCH', '1mCL', '1mCV'}; 
    elseif s == 3
        sessions = {'PreRest', '1mRest', '3mRest', '1mCH', '3mCH', '1mCL', '3mCL', '1mCV', '3mCV'}; 
    elseif s == 8
        sessions = {'PreRest', '3mRest', '6mRest', '12mRest', '3mCH', '6mCH', '12mCH', '3mCL', '6mCL', '12mCL', '3mCV', '6mCV', '12mCV'}; 
    else 
        sessions = {'PreRest', '1mRest', '3mRest', '6mRest', '12mRest', '1mCH', '3mCH', '6mCH', '12mCH', '1mCL', '3mCL', '6mCL', '12mCL', '1mCV', '3mCV', '6mCV', '12mCV'}; 
    end

    for i = 1:length(sessions)
        sess = sessions{i};      
        load([inpath subj '/' sess '/FC_matrix_test_r.mat']);
        FC_matrix_test(i,:,:) = CorrMat_test;
        load([inpath subj '/' sess '/FC_matrix_retest_r.mat']);
        FC_matrix_retest(i,:,:) = CorrMat_retest;
    end
    
    % Get the upper triangle of the test matrix as a vector
    CorrMat = squeeze(mean(FC_matrix_test));
    CorrMat(isnan(CorrMat)) = 0;
    k = 0;
    for i = 1:num
        for j = i+1:num
            k = k + 1;
            testMatrix(k, s) = CorrMat(i, j);
        end
    end

    % Get the upper triangle of the retest matrix as a vector
    CorrMat = squeeze(mean(FC_matrix_retest));
    CorrMat(isnan(CorrMat)) = 0;
    k = 0;
    for i = 1:num
        for j = i+1:num
            k = k + 1;
            retestMatrix(k, s) = CorrMat(i, j);
        end
    end
end

% Compute the average of test and retest matrices
Com_FC_matrix = (testMatrix + retestMatrix) / 2;

% Compute intra-session similarity
Intra_Simi = diag(corr(testMatrix, retestMatrix))';
Intra_Simi_matrix = corr(testMatrix, retestMatrix);

% Save the results as CSV files
writematrix(Intra_Simi_matrix, [outpath '/Intra_Simi_152matrix.csv']);
