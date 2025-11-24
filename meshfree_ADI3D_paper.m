function R = meshfree_ADI3D_paper(iSub, rootDir, useGPU)
% Fast paper-aligned ADI -> DoD -> adaptive Otsu -> FCM([FLAIR,DoD])
% Usage: R = meshfree_ADI3D_paper_fast(i, rootDir, useGPU)

if nargin < 3, useGPU = true; end

%% ---- Fixed params (safe defaults) ----
alpha   = 1.40;     % fractional order
lambda  = 0.35;     % diffusion strength
t1      = 4;
t2      = 12;
neigh   = 3;
fusePlanes = false; % turn OFF for speed; enable later if needed
resizeTo   = [];    % [] = full resolution


%% ---- Discover files (do NOT override inputs) ----
imgDir = fullfile(rootDir,'imagesTr');
lblDir = fullfile(rootDir,'labelsTr');
imgs = [dir(fullfile(imgDir,'*.nii*')); dir(fullfile(imgDir,'*.nii.gz*'))];
lbls = [dir(fullfile(lblDir,'*.nii*')); dir(fullfile(lblDir,'*.nii.gz*'))];

assert(~isempty(imgs) && ~isempty(lbls), ...
    'No NIfTI files found. Check %s and %s', imgDir, lblDir);

[~,ix]=sort({imgs.name}); imgs=imgs(ix);
[~,jx]=sort({lbls.name}); lbls=lbls(jx);
assert(iSub>=1 && iSub<=numel(imgs) && iSub<=numel(lbls), 'iSub out of range.');

fprintf('>>> Case %d: %s\n', iSub, imgs(iSub).name);

%% ---- Load FLAIR & SEG ----
I  = double(niftiread(niftiinfo(fullfile(imgs(iSub).folder, imgs(iSub).name))));
GT = double(niftiread(niftiinfo(fullfile(lbls(iSub).folder, lbls(iSub).name)))) > 0;

if ~isempty(resizeTo)
    I  = imresize3(I, resizeTo, 'linear');
    GT = imresize3(GT, resizeTo, 'nearest');
end

%% ---- Normalize inside brain (robust) ----
brain = (I > 0) & isfinite(I);
if ~any(brain(:)), brain = isfinite(I); end
lo = prctile(I(brain), 1);
hi = prctile(I(brain),99);
I = (I - lo) / (hi - lo + eps);
I = max(0, min(1, I));

%% ---- Fractional 1-D kernels ----
r = (-neigh:neigh)'; r(r==0)=1e-8;
k = (abs(r).^(-alpha)) / gamma(1-alpha);
k = k / sum(abs(k));
kx = reshape(k,[],1,1); ky = reshape(k,1,[],1); kz = reshape(k,1,1,[]);

%% ---- GPU option ----
canGPU = useGPU && gpuDeviceCount>0;
if canGPU
    I  = gpuArray(single(I));
    kx = gpuArray(single(kx)); ky = gpuArray(single(ky)); kz = gpuArray(single(kz));
end

%% ---- ADI to t1 and t2 ----
u = I;
for n=1:t1
    u = sweep(u,kx,ky,kz,lambda);
end
u1 = u;
for n=1:(t2-t1)
    u = sweep(u,kx,ky,kz,lambda);
end
u2 = u;

if canGPU, I=gather(I); u1=gather(u1); u2=gather(u2); end

DoD = abs(u2 - u1);
DoD = (DoD - min(DoD(:))) / (max(DoD(:)) - min(DoD(:)) + eps);

%% ---- Blockwise adaptive Otsu ----
sz = size(DoD);
bx=max(round(sz(1)/4),8); by=max(round(sz(2)/4),8); bz=max(round(sz(3)/3),4);
gTh = graythresh(DoD);
bw  = false(sz);

for x=1:bx:sz(1)
    x2=min(x+bx-1,sz(1));
    for y=1:by:sz(2)
        y2=min(y+by-1,sz(2));
        for z=1:bz:sz(3)
            z2=min(z+bz-1,sz(3));
            blk = DoD(x:x2,y:y2,z:z2);
            if ~any(blk(:)), continue; end
            tloc = graythresh(blk);
            th = 0.7*tloc + 0.3*gTh;
            bw(x:x2,y:y2,z:z2) = blk > th;
        end
    end
end
bw = imopen(imclose(bw, ones(3,3,3)), ones(3,3,3));
bw = bwareaopen(bw, round(numel(bw)*2e-4));

%% ---- FCM on candidate voxels only (FAST) ----
featAll = [I(:), DoD(:)];
roi = bw(:)==1;
X = double(featAll(roi,:));

if isempty(X)
    mask = bw; % nothing to refine
else
    if exist('fcm','file')==2
        fprintf('   FCM on %d voxels...\n', size(X,1));
        [C,U] = fcm(X, 3, [2 100 1e-5 0]);
        [~,h] = max(U,[],1); h = h(:);
        [~,tumLab] = max(C(:,2));  % highest DoD
    else
        fprintf('   fcm() not found -> using kmeans.\n');
        [h,~] = kmeans(X, 3, 'MaxIter',200, 'Replicates',2);
        [~,tumLab] = max(accumarray(h, X(:,2), [], @mean));
    end
    mask = false(numel(roi),1);
    mask(roi) = (h == tumLab);
    mask = reshape(mask, sz);
end

mask = imopen(imclose(mask, ones(3,3,3)), ones(3,3,3));

%% ---- Optional 3-plane fusion (off by default for speed) ----
if fusePlanes
    mask = fuse_3views(I,mask,alpha,lambda,t1,t2);
end

%% ---- Largest 3D CC ----
CC = bwconncomp(mask,26);
if CC.NumObjects>0
    np = cellfun(@numel, CC.PixelIdxList);
    [~,id] = max(np);
    keep = false(size(mask)); keep(CC.PixelIdxList{id}) = true;
    mask = keep;
end

%% ---- Metrics (whole tumor) ----
TP = nnz(mask & GT); FP = nnz(mask & ~GT);
FN = nnz(~mask & GT); TN = nnz(~mask & ~GT);
Dice = 2*TP/(2*TP + FP + FN + eps);
Jacc = TP/(TP + FP + FN + eps);
Sens = TP/(TP + FN + eps);
Spec = TN/(TN + FP + eps);
Acc  = (TP + TN) / numel(GT);
HD95 = hd95(mask, GT);

R.metrics = struct('Dice',Dice,'Jacc',Jacc,'Sens',Sens,'Spec',Spec,'Acc',Acc,'HD95',HD95);
R.mask = mask;
end

%% ===== Helpers =====
function u = sweep(u,kx,ky,kz,lambda)
u = u + lambda*convn(u,kx,'same');
u = u + lambda*convn(u,ky,'same');
u = u + lambda*convn(u,kz,'same');
end

function h = hd95(A,B)
if ~any(A(:)) && ~any(B(:)), h=0;  return; end
if ~any(A(:)) || ~any(B(:)), h=inf; return; end
SE = strel('sphere',1);
As = A & ~imdilate(~A,SE);
Bs = B & ~imdilate(~B,SE);
DA = bwdist(~As);
DB = bwdist(~Bs);
h = prctile([DB(As); DA(Bs)], 95);
end

function M = fuse_3views(I, M, alpha, lambda, t1, t2)
% Robust (safe) multi-plane fusion; slower – leave OFF unless needed
sz = size(I); M2 = M;
% Axial
for k=1:sz(3)
    bw = seg2D(I(:,:,k), alpha, lambda, t1, t2);
    if ~isequal(size(bw), [sz(1) sz(2)]), bw = imresize(bw,[sz(1) sz(2)],'nearest'); end
    M2(:,:,k) = M2(:,:,k) | bw;
end
% Coronal
for k=1:sz(2)
    bw = seg2D(squeeze(I(:,k,:)), alpha, lambda, t1, t2);
    if ~isequal(size(bw), [sz(1) sz(3)]), bw = imresize(bw,[sz(1) sz(3)],'nearest'); end
    for x=1:sz(1), for z=1:sz(3), M2(x,k,z) = M2(x,k,z) | bw(x,z); end, end
end
% Sagittal
for k=1:sz(1)
    bw = seg2D(squeeze(I(k,:,:)), alpha, lambda, t1, t2);
    if ~isequal(size(bw), [sz(2) sz(3)]), bw = imresize(bw,[sz(2) sz(3)],'nearest'); end
    for y=1:sz(2), for z=1:sz(3), M2(k,y,z) = M2(k,y,z) | bw(y,z); end, end
end
M = imopen(imclose(M2, ones(3,3,3)), ones(3,3,3));
M = bwareaopen(M, round(numel(M)*1e-4));
end

function BW = seg2D(im,alpha,lambda,t1,t2)
im = (im-min(im(:))) / (max(im(:))-min(im(:)) + eps);
r = -3:3; r(r==0)=1e-8;
k = (abs(r).^(-alpha))/gamma(1-alpha); k = k/sum(abs(k));
u = im;
for n=1:t1, u = u + lambda*conv2(u,k,'same');  u = u + lambda*conv2(u,k','same'); end
u1 = u;
for n=1:(t2-t1), u = u + lambda*conv2(u,k,'same');  u = u + lambda*conv2(u,k','same'); end
DoD = abs(u - u1); DoD = (DoD-min(DoD(:)))/(max(DoD(:))-min(DoD(:))+eps);
BW = DoD > graythresh(DoD);
BW = imclose(imopen(BW,ones(3)),ones(3));
BW = bwareaopen(BW, max(20, round(numel(BW)*0.002)));
BW = logical(BW);
end