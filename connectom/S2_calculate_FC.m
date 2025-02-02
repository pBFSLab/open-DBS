clear; clc
addpath('./utilities') % Add utilities at first
% addpath('./Visualization')

conf = 2; % Confidence value threshold: a parameter to control the percentage
% of individual info, lower means more individual. Suggested values: 1.3, 2, 3.
% Default is 3. For data with long scanning times, it is recommended to use a lower value.
confs = num2str(conf);
CalAtlas = 0;
masknames = {'Visual', 'Motor', 'Frontal', 'Temporal', 'Parietal'};
bset_clusternum_lh = [12, 12, 17, 20, 15];
bset_clusternum_rh = bset_clusternum_lh;
alternatives = {[6, 6, 13, 10, 11], [12, 12, 17, 20, 15], [16, 21, 17, 30, 24]};

trace_alternatives = {'PreRest', '1mRest', '1mCH', '1mCV', '1mCL', '3mRest', '3mCH', '3mCV', '3mCL', '6mRest', '6mCH', '6mCV', '6mCL', '12mRest', '12mCH', '12mCV', '12mCL'};

DataPath = './data/open_DBS/App_data/DBS'; % Data path (relative path)
OutPath = './data/open_DBS/Parcellation152'; % Output path (relative path)
mkdir(OutPath);

sublist = './docs/DBS_subject_list.txt'; % Subject list (relative path)

subs = textread(sublist, '%s');
tic;
for s = 1:length(subs)
    sub = subs{s};
    disp(['Loading Data: ' sub ' ...'])
    SubDataPath = dir([DataPath, '/', sub, '/*/Preprocess']);

    % Extract all folders
    SubDataFolder = {SubDataPath.folder};
    % Remove duplicates
    SubDataFolder = unique(SubDataFolder);

    traces_processed = [];

    %% Read individual parcellation results
    [vertex_l, label_l, ct_l] = read_annotation([OutPath '/WB_lh/' sub '_Combo/' 'Combine/' 'lh.' sub '_Combo_IndiCluster76_parc_result_fs6.annot']);
    [vertex_r, label_r, ct_r] = read_annotation([OutPath '/WB_rh/' sub '_Combo/' 'Combine/' 'rh.' sub '_Combo_IndiCluster76_parc_result_fs6.annot']);

    % Process each folder
    for i = 1:length(SubDataFolder)
        % Check if the folder contains a surf subfolder
        if ~exist([SubDataFolder{i} '/surf'], 'dir')
            continue
        end
        % Get the corresponding trace from the trace_alternatives
        [foldername, ~] = fileparts(SubDataFolder{i});
        if sum(cellfun(@(x) contains(foldername, x), trace_alternatives)) == 0
            disp(['   Folder ', foldername, ' does not contain a valid trace'])
            continue
        end
        trace = trace_alternatives{cellfun(@(x) contains(foldername, x), trace_alternatives)};
        sub_trace = [sub, '_', trace];
        if ismember(sub_trace, traces_processed)
            disp(['   Trace ', sub_trace, ' has another same named session'])
            continue
        end
        traces_processed{end+1} = sub_trace;

        % Check if already processed
        % if exist([OutPath '/' masknames{5} '_lh/IndiPar/Iters_' confs '_Cluster' num2str(bset_clusternum_lh(5)) '/seg2/' sub_trace '/Iter_10/Network_1_lh.mgh'], 'file')
        %     fprintf('   Skip the processed trace %s\n', sub_trace)
        %     continue
        % end
        
        lhData = single(ConcatenateSurf(SubDataFolder{i}, sub, 'lh', 'fsaverage6'));
        rhData = single(ConcatenateSurf(SubDataFolder{i}, sub, 'rh', 'fsaverage6'));

        %% Split data into two parts for test and retest
        test_data_lh = lhData(:, 1:round(size(lhData, 2)/2));
        test_data_rh = rhData(:, 1:round(size(rhData, 2)/2));

        retest_data_lh = lhData(:, round(size(lhData, 2)/2)+1:end);
        retest_data_rh = rhData(:, round(size(rhData, 2)/2)+1:end);

        % Extract mean time series for parcellation
        % Initialize surf_indi_mean_test and surf_indi_mean_retest
        surf_lh_indi_mean_test = zeros(size(test_data_lh, 2), 76);
        surf_rh_indi_mean_test = zeros(size(test_data_rh, 2), 76);

        surf_lh_indi_mean_retest = zeros(size(retest_data_lh, 2), 76);
        surf_rh_indi_mean_retest = zeros(size(retest_data_rh, 2), 76);

        for z = 1:76
            [indi_lh, n_lh] = find(label_l == ct_l.table(z+1, 5));
            [indi_rh, n_rh] = find(label_r == ct_r.table(z+1, 5));

            %% Test
            surf_lh_indi_mean_test(:, z) = nanmean(test_data_lh(indi_lh, :), 1);
            surf_rh_indi_mean_test(:, z) = nanmean(test_data_rh(indi_rh, :), 1);

            %% Retest
            surf_lh_indi_mean_retest(:, z) = nanmean(retest_data_lh(indi_lh, :), 1);
            surf_rh_indi_mean_retest(:, z) = nanmean(retest_data_rh(indi_rh, :), 1);
        end

        surf_indi_mean_test = [surf_lh_indi_mean_test, surf_rh_indi_mean_test];
        surf_indi_mean_retest = [surf_lh_indi_mean_retest, surf_rh_indi_mean_retest];

        OutPath_sub = [OutPath '/' 'Indi_FC/' sub '/' trace];
        if ~exist(OutPath_sub, 'dir')
            mkdir(OutPath_sub)
        end

        % Test
        CorrMat_test = corrcoef(surf_indi_mean_test);
        CorrMat_test(isnan(CorrMat_test)) = 0;
        %% Save test data
        save([OutPath_sub '/FC_matrix_test_r.mat'], 'CorrMat_test');
        CorrMat_test = atanh(CorrMat_test);
        CorrMat_test(isnan(CorrMat_test)) = 0;
        CorrMat_test(isinf(CorrMat_test)) = 0;
        save([OutPath_sub '/FC_matrix_test_z.mat'], 'CorrMat_test');

        % Retest
        CorrMat_retest = corrcoef(surf_indi_mean_retest);
        CorrMat_retest(isnan(CorrMat_retest)) = 0;
        %% Save retest data
        save([OutPath_sub '/FC_matrix_retest_r.mat'], 'CorrMat_retest');
        CorrMat_retest = atanh(CorrMat_retest);
        CorrMat_retest(isnan(CorrMat_retest)) = 0;
        CorrMat_retest(isinf(CorrMat_retest)) = 0;
        save([OutPath_sub '/FC_matrix_retest_z.mat'], 'CorrMat_retest');

        % break % For testing, only run one folder
    end
    % break % For testing, only run one subject
end
