clear all; clc;

% Read the data saved in the csv file
cbeta_GP_csv = readtable('./data/cbeta_mean_GPi.csv'); % Relative path

%% Extract 'test' and 'retest' columns
cbeta_GP = cbeta_GP_csv{:,3:4};

%% Transpose the data
cbeta_GP = cbeta_GP';

%% Add a dimension of ones in the first dimension
cbeta_GP = permute(cbeta_GP,[3 1 2]);

c = cbeta_GP_csv{:,6};
s = cbeta_GP_csv{:,5};

tbl_DBS = table(reshape(cbeta_GP(:,1,:),[],1), reshape(cbeta_GP(:,2,:),[],1), num2str(c(:),'%02d'), num2str(s(:),'%02d'), 'VariableNames', {'UPDRS','fMRI','Cond','Subject'});

LME1 = fitlme(tbl_DBS, 'UPDRS ~ 1 + (1 |Subject)', 'FitMethod','REML');
LME2 = fitlme(tbl_DBS, 'fMRI ~ 1 + (1 | Subject)', 'FitMethod','REML');

zupdrs_ = double(residuals(LME1,'residualtype','standardized'));
zfmri_ = double(residuals(LME2,'residualtype','standardized'));

[rc,p] = corr(zfmri_,zupdrs_,'row','pairwise');

inter = var(mean([zfmri_ zupdrs_],2,'omitnan'),'omitnan');
intra = mean(var([zfmri_ zupdrs_],[],2,'omitnan'),'omitnan');
icc = inter / (inter + intra);

[r, g, b] = ndgrid([0 0.5 1], [0 0.5 1], [0 0.5 1]);
r = r(:); r = r(2:end,1);
g = g(:); g = g(2:end,1);
b = b(:); b = b(2:end,1);

% Use new color palette
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

r = colors(:,1);
g = colors(:,2);
b = colors(:,3);

figure;

% Plot scatter points
for i = 1:14
    plot(zfmri_(s(:)==i), zupdrs_(s(:)==i), '.', 'color', [r(i) g(i) b(i)], 'markersize', 40, 'linewidth', 5);
    hold on;
end

% Plot fitting lines
for i = 1:14
    regx = [zfmri_(s(:)==i)];
    y = zupdrs_(s(:)==i);
    mask = ~isnan(regx) & ~isnan(y);
    regx = [regx(mask,:) ones(sum(mask), 1)];
    y = y(mask,:);
    if sum(mask) > 2
        beta = (regx.' * regx) \ regx.' * y;
        plot([min(regx(:,1)) max(regx(:,1))], [min(regx(:,1)) 1; max(regx(:,1)) 1] * beta, '-', 'color', [r(i) g(i) b(i)], 'linewidth', 2);
        hold on;
    end
end

regx = [zfmri_(:)];
y = zupdrs_(:);
mask = ~isnan(regx) & ~isnan(y);
regx = [regx(mask,:) ones(sum(mask), 1)];
y = y(mask,:);

beta = (regx.' * regx) \ regx.' * y;

hold on;

lim_f = floor(min(min([regx(:,1) y(:,1)]))/3) * 3;
lim_u = ceil(max(max([regx(:,1) y(:,1)]))/3) * 3;

box off;
axis([-4 4 -4 4]);
xticks([-100:2:100]); yticks([-100:2:100]);
xticklabels([]); yticklabels([]);
set(gca,'tickdir','out', 'ticklength', [.02 .02]);
set(gca,'linewidth',3);
set(gca,'fontname','arial','fontsize',25);
set(gcf,'position', [0 0 1000 800]);
set(gcf,'color', 'white');

print -dpng -r600 -noui Reliability/GP_inter_subject2.png;

print -depsc2 -painters test.eps;
eval(['!mv test.eps Reliability/GP_inter_subject.eps']);
close all;

% Read the M1 data
cbeta_M1_csv = readtable('./data/cbeta_mean_M1.csv'); % Relative path
cbeta_M1 = cbeta_M1_csv{:,3:4};
cbeta_M1 = cbeta_M1';
cbeta_M1 = permute(cbeta_M1,[3 1 2]);

c = cbeta_M1_csv{:,6};
s = cbeta_M1_csv{:,5};

tbl_DBS = table(reshape(cbeta_M1(:,1,:),[],1), reshape(cbeta_M1(:,2,:),[],1), num2str(c(:),'%02d'), num2str(s(:),'%02d'), 'VariableNames', {'UPDRS','fMRI','Cond','Subject'});

LME1 = fitlme(tbl_DBS, 'UPDRS ~ 1 + Subject', 'FitMethod','REML');
LME2 = fitlme(tbl_DBS, 'fMRI ~ 1 + Subject', 'FitMethod','REML');

zupdrs_ = residuals(LME1, 'residualtype', 'standardized');
zfmri_ = residuals(LME2, 'residualtype', 'standardized');

[rc, p] = corr(zfmri_, zupdrs_, 'row', 'pairwise');

inter = var(mean([zfmri_ zupdrs_], 2, 'omitnan'), 'omitnan');
intra = mean(var([zfmri_ zupdrs_], [], 2, 'omitnan'), 'omitnan');
icc = inter / (inter + intra);

figure;

% Plot scatter points for M1
for i = 1:14
    plot(zfmri_(s(:)==i), zupdrs_(s(:)==i), '.', 'color', [r(i) g(i) b(i)], 'markersize', 40, 'linewidth', 5);
    hold on;
end

% Plot fitting lines for M1
for i = 1:14
    regx = [zfmri_(s(:)==i)];
    y = zupdrs_(s(:)==i);
    mask = ~isnan(regx) & ~isnan(y);
    regx = [regx(mask,:) ones(sum(mask), 1)];
    y = y(mask,:);
    if sum(mask) > 2
        beta = (regx.' * regx) \ regx.' * y;
        plot([min(regx(:,1)) max(regx(:,1))], [min(regx(:,1)) 1; max(regx(:,1)) 1] * beta, '-', 'color', [r(i) g(i) b(i)], 'linewidth', 2);
        hold on;
    end
end

regx = [zfmri_(:)];
y = zupdrs_(:);
mask = ~isnan(regx) & ~isnan(y);
regx = [regx(mask,:) ones(sum(mask), 1)];
y = y(mask,:);

beta = (regx.' * regx) \ regx.' * y;

hold on;

lim_f = floor(min(min([regx(:,1) y(:,1)]))/3) * 3;
lim_u = ceil(max(max([regx(:,1) y(:,1)]))/3) * 3;

box off;
axis([-4 4 -4 4]);
xticks([-100:2:100]); yticks([-100:2:100]);
xticklabels([]); yticklabels([]);
set(gca,'tickdir','out', 'ticklength', [.02 .02]);
set(gca,'linewidth',3);
set(gca,'fontname','arial','fontsize',25);
set(gcf,'position', [0 0 1000 800]);
set(gcf,'color', 'white');

print -dpng -r600 -noui Reliability/M1_inter_subject2.png;

print -depsc2 -painters test.eps;
eval(['!mv test.eps Reliability/M1_inter_subject.eps']);
close all;
