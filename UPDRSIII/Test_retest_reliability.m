% clear all; clc;
% This script computes the test-retest reliability of UPDRS-III scores

% Load data containing various sub-scores
all_data = readtable('./data/UPDRSIII_13patients.csv');
% Adjust subject IDs in the data (add 2)
all_data.patient = all_data.patient + 2;

% Load pre-test data
pre_data = readtable('./data/UPDRSIII_pre_14patients.csv');
% Adjust subject IDs in pre-test data (add 1)
pre_data.patient = pre_data.patient + 1;

% Merge data
all_data = [all_data; pre_data];

% Filter data where measurement is 'total'
data = all_data(strcmp(all_data.measurement, 'total'), :);
data.Properties.VariableNames{'patient'} = 'Subject';

s = data.Subject;

LME1 = fitlme(data, 'score ~ 1 + (1 |Subject)', 'FitMethod', 'REML');
LME2 = fitlme(data, 'score_retest ~ 1 + (1 | Subject)', 'FitMethod', 'REML');

zupdrs_ = double(residuals(LME1, 'residualtype', 'standardized'));
zupdrs_2 = double(residuals(LME2, 'residualtype', 'standardized'));

[rc, p] = corr(zupdrs_2, zupdrs_, 'row', 'pairwise');

inter = var(mean([zupdrs_2 zupdrs_], 2, 'omitnan'), 'omitnan');
intra = mean(var([zupdrs_2 zupdrs_], [], 2, 'omitnan'), 'omitnan');
icc = inter / (inter + intra);

% Define new color palette
colors = [
    0.6396, 0.4000, 0.4000; % #a36665
    0.7412, 0.3255, 0.4392; % #bd5370
    0.8275, 0.5490, 0.4745; % #d38c79
    0.7102, 0.6941, 0.4863; % #b5b17b
    0.9725, 0.6549, 0.5569; % #f8a78e
    0.5533, 0.6824, 0.5020; % #8daf80
    0.6941, 0.8314, 0.7333; % #b1d4bb
    0.9608, 0.9490, 0.8000; % #f5f2cc
    0.4667, 0.4824, 0.6941; % #777ab1
    0.5725, 0.4078, 0.5686; % #926891
    0.9137, 0.6510, 0.7137; % #e9a6b6
    0.3804, 0.7020, 0.7020; % #61b3b3
    0.7059, 0.7059, 0.7059; % #b4b4b4
    0.7373, 0.8275, 0.8510   % #bcd3d9
];
r = colors(:, 1);
g = colors(:, 2);
b = colors(:, 3);

zupdrs_ = data.score;
zupdrs_2 = data.score_retest;

figure;

% Plot scatter for each subject
for i = 1:14
    if length(zupdrs_2(s(:) == i)) < 2
        continue;
    end
    plot(zupdrs_2(s(:) == i), zupdrs_(s(:) == i), '.', 'color', [r(i) g(i) b(i)], 'markersize', 40, 'linewidth', 5);
    hold on;
end

% Plot regression lines for each subject
for i = 1:14
    if length(zupdrs_2(s(:) == i)) < 2
        continue;
    end
    regx = [zupdrs_2(s(:) == i)];
    y = zupdrs_(s(:) == i);
    mask = ~isnan(regx) & ~isnan(y);
    regx = [regx(mask, :) ones(sum(mask), 1)];
    y = y(mask, :);
    if sum(mask) > 2
        beta = (regx.' * regx) \ regx.' * y;
        plot([min(regx(:, 1)) max(regx(:, 1))], [min(regx(:, 1)) 1; max(regx(:, 1)) 1] * beta, '-', 'color', [r(i) g(i) b(i)], 'linewidth', 2);
        hold on;
    end
end

regx = [zupdrs_2(:)];
y = zupdrs_(:);
mask = ~isnan(regx) & ~isnan(y);
regx = [regx(mask, :) ones(sum(mask), 1)];
y = y(mask, :);

beta = (regx.' * regx) \ regx.' * y;
hold on;

lim_f = floor(min(min([regx(:, 1) y(:, 1)])) / 3) * 3;
lim_u = ceil(max(max([regx(:, 1) y(:, 1)])) / 3) * 3;

box off;
axis([0 80 0 80]);
xticks([-100:20:100]); yticks([-100:20:100]);
xticklabels([]); yticklabels([]);
set(gca, 'tickdir', 'out', 'ticklength', [.02 .02]);
set(gca, 'linewidth', 3);
set(gca, 'fontname', 'arial', 'fontsize', 25);
set(gcf, 'position', [0 0 1000 800]);
set(gcf, 'color', 'white');

% Save the figure as PNG
print('-dpng', '-r600', '-noui', './Reliability/UPDRS_line_w2.png');
print -depsc2 -painters test.eps;
eval(['!mv test.eps ./Reliability/UPDRS_line_w2.eps']);
close all;

%% Repeat the same steps for different sub-scores
measurement_list = {'tremor', 'rigidity', 'bradykinesia', 'gait', 'posture', 'PIGD'};
axis_max_list = {12, 20, 40, 5, 15, 20};
ticks_interval_list = {3, 5, 10, 1, 5, 5};

% Load data containing different sub-scores
all_data = readtable('./data/UPDRSIII_13patients.csv');
% Adjust subject IDs in the data (add 2)
all_data.patient = all_data.patient + 2;

% Load pre-test data
pre_data = readtable('./data/UPDRSIII_14patients_pre.csv');
% Adjust subject IDs in pre-test data (add 1)
pre_data.patient = pre_data.patient + 1;

% Merge data
all_data = [all_data; pre_data];

%% Extract data for each sub-score, compute ICC, and plot
for i = 1:length(measurement_list)
    measurement = measurement_list{i};
    % Filter data for the given measurement
    data = all_data(strcmp(all_data.measurement, measurement), :);
    data.Properties.VariableNames{'patient'} = 'Subject';
    
    % Save data as CSV
    writetable(data, ['./Reliability/UPDRS_' measurement '.csv']);

    s = data.Subject;

    LME1 = fitlme(data, 'score ~ 1 + (1 |Subject)', 'FitMethod', 'REML');
    LME2 = fitlme(data, 'score_retest ~ 1 + (1 | Subject)', 'FitMethod', 'REML');

    zupdrs_ = double(residuals(LME1, 'residualtype', 'standardized'));
    zupdrs_2 = double(residuals(LME2, 'residualtype', 'standardized'));

    [rc, p] = corr(zupdrs_2, zupdrs_, 'row', 'pairwise');

    inter = var(mean([zupdrs_2 zupdrs_], 2, 'omitnan'), 'omitnan');
    intra = mean(var([zupdrs_2 zupdrs_], [], 2, 'omitnan'), 'omitnan');
    icc = inter / (inter + intra);

    % Set axis limits for each sub-score
    axis_max = axis_max_list{i};
    ticks_interval = ticks_interval_list{i};

    % Plot for each subject
    figure;
    for i = 1:14
        if length(zupdrs_2(s(:) == i)) < 2
            continue;
        end
        plot(zupdrs_2(s(:) == i), zupdrs_(s(:) == i), '.', 'color', [r(i) g(i) b(i)], 'markersize', 40, 'linewidth', 5);
        hold on;
    end

    % Regression lines
    for i = 1:14
        if length(zupdrs_2(s(:) == i)) < 2
            continue;
        end
        regx = [zupdrs_2(s(:) == i)];
        y = zupdrs_(s(:) == i);
        mask = ~isnan(regx) & ~isnan(y);
        regx = [regx(mask, :) ones(sum(mask), 1)];
        y = y(mask, :);
        if sum(mask) > 2
            beta = (regx.' * regx) \ regx.' * y;
            plot([min(regx(:, 1)) max(regx(:, 1))], [min(regx(:, 1)) 1; max(regx(:, 1)) 1] * beta, '-', 'color', [r(i) g(i) b(i)], 'linewidth', 2);
            hold on;
        end
    end

    regx = [zupdrs_2(:)];
    y = zupdrs_(:);
    mask = ~isnan(regx) & ~isnan(y);
    regx = [regx(mask, :) ones(sum(mask), 1)];
    y = y(mask, :);

    beta = (regx.' * regx) \ regx.' * y;
    hold on;

    box off;
    axis([0 axis_max 0 axis_max]);
    xticks([0:ticks_interval:axis_max]);
    yticks([0:ticks_interval:axis_max]);
    xticklabels([]); yticklabels([]);
    set(gca, 'tickdir', 'out', 'ticklength', [.02 .02]);
    set(gca, 'linewidth', 3);
    set(gca, 'fontname', 'arial', 'fontsize', 25);
    set(gcf, 'position', [0 0 1000 800]);
    set(gcf, 'color', 'white');

    % Save the figure as PNG
    print('-dpng', '-r600', '-noui', ['./Reliability/UPDRS_' measurement '_line_w2.png']);
    print -depsc2 -painters test.eps;
    eval(['!mv test.eps ./Reliability/UPDRS_' measurement '_line_w2.eps']);
    close all;
end
