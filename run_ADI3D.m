%% ===================== RUN ADI 3D BATCH FAST ======================
clear; clc; close all;

% ✅ Change this path to your ADI-prepared BraTS folder
rootDir = 'K:\explo_parv';

% Results output directory
outDir = fullfile(rootDir, "results_ADI_fast");
if ~exist(outDir, 'dir'), mkdir(outDir); end

% Collect subject files
imgs = dir(fullfile(rootDir, 'imagesTr', "*.nii*"));
[~,ix] = sort({imgs.name}); imgs = imgs(ix);
N = min(369, numel(imgs));
fprintf("Found %d subjects\n\n", N);

% Pre-allocate results
Dice  = nan(N,1);
Jacc  = nan(N,1);
Sens  = nan(N,1);
Spec  = nan(N,1);
Acc   = nan(N,1);
HD95  = nan(N,1);

%% ===================== Loop All Subjects ======================
tic;
for i = 1:N
    fprintf(">>> [%3d / %3d] %s ... ", i, N, imgs(i).name);

    try
        R = meshfree_ADI3D_paper(i, rootDir, true);  % GPU ON
        Dice(i) = R.metrics.Dice;
        Jacc(i) = R.metrics.Jacc;
        Sens(i) = R.metrics.Sens;
        Spec(i) = R.metrics.Spec;
        Acc(i)  = R.metrics.Acc;
        HD95(i) = R.metrics.HD95;

        fprintf("Dice=%.3f | HD95=%.1f\n", Dice(i), HD95(i));

    catch ME
        fprintf("FAILED ❌  (%s)\n", ME.message);
        Dice(i) = NaN;  % Keep NaN for missing cases
    end
end
runtime = toc;

fprintf("\n✅ Batch complete in %.1f minutes\n", runtime/60);

%% ===================== Save Results ======================
T = table(Dice, Jacc, Sens, Spec, Acc, HD95);
csvPath = fullfile(outDir, "ADI_metrics.csv");
writetable(T, csvPath);
fprintf("📁 Saved metrics: %s\n", csvPath);

try
    writetable(T, fullfile(outDir, "ADI_metrics.xlsx"));
catch
    fprintf("⚠ Excel not available — only CSV saved.\n");
end

%% ===================== Plot Metrics ======================
valid = Dice(~isnan(Dice) & Dice>0 & Dice<=1);

if isempty(valid)
    warning("⚠ No valid Dice values — skipping plots.");
else
    %% Histogram + KDE
    fig1 = figure('Color','w'); 
    histogram(valid, 15,'Normalization','pdf'); hold on;
    try
        [fp,xp] = ksdensity(valid);
        plot(xp,fp,'LineWidth',2);
    catch
        warning("ksdensity failed — histogram only");
    end
    title('Dice Distribution'); xlabel('Dice'); ylabel('Density');
    grid on; exportgraphics(fig1, fullfile(outDir,'Dice_Histogram.png'), 'Resolution',200);

    %% Box & Violin
    fig2 = figure('Color','w');
    subplot(1,2,1); boxplot(valid); title('Dice Boxplot'); grid on;
    subplot(1,2,2);
    if exist('violinplot','file')==2
        violinplot(valid); title('Dice Violin'); grid on;
    else
        % fallback mini-violin
        [fv,xv] = ksdensity(valid); fv=fv/max(fv)*0.3;
        patch([1-fv, fliplr(1+fv)], [xv, fliplr(xv)], [0.3 0.6 0.9], 'EdgeColor','none');
        xlim([0.5 1.5]); set(gca,'XTick',[]); title('Dice Violin (fallback)'); grid on;
    end
    exportgraphics(fig2, fullfile(outDir,'Dice_Box_Violin.png'), 'Resolution',200);
end

%% ===================== Summary Stats ======================
if ~isempty(valid)
    fprintf("\n📊 Mean Dice:   %.3f\n", mean(valid));
    fprintf("🟢 Median Dice: %.3f\n", median(valid));
    fprintf("📌 Q1–Q3:       %.3f – %.3f\n", quantile(valid,0.25), quantile(valid,0.75));
else
    fprintf("\n⚠ No valid metric values found.\n");
end

fprintf("\n🎯 DONE\n");