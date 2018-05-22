%% generate deconv pics
clear;clc;
addpath('/home/zhongyaoyao/caffe-master/matlab'); % change your path
caffe.reset_all();
net_weights = ['/home/zhongyaoyao/caffe-master/examples/vggface2/models/vggface/VGG_FACE.caffemodel'];
net_model = ['/home/zhongyaoyao/caffe-master/examples/vggface2/models/vggface/VGG_FACE_deploy_drelu.prototxt'];
caffe.set_mode_gpu();
net = caffe.Net(net_model, net_weights, 'test'); 

load('listp.mat');
mkdir('pic5000_pool5result');
averageImage = [129.1863,104.7624,93.5940]; 
IMAGE_DIM = 256;
CROPPED_DIM = 224;
for featuremap=22:22
    featuremap
    tic;
    load(['layerpic5000pick_pool5_feature' num2str(featuremap) '.mat']); % load the sort of activations
    pic=picsort{1}(1:50); % top 50 pics of a neuron 
for ii = 1:50
    % preproccess the image
    img0 = imread(['ppic5000pick/' num2str(pic(ii)) '.jpg']);
    img = single(img0);
    img = cat(3,img(:,:,1)-averageImage(1),...
        img(:,:,2)-averageImage(2),...
        img(:,:,3)-averageImage(3));
    img = img(:, :, [3, 2, 1]); 
    img = permute(img, [2, 1, 3]); % permute width and height
    img = imresize(img, [IMAGE_DIM IMAGE_DIM], 'bilinear');  
    img_crops(:,:,:,1)=img(17:240,17:240,:);
    net.blobs('data').set_data(img_crops);
    % forward
    net.forward_prefilled();
    pooled=net.blobs('pool5').get_data();
    % select the maximum activation
    pooledSelect=zeros(size(pooled));  
    selectmap=pooled(:,:,featuremap);
    [value,id]=sort(selectmap(:),'descend');
    selectmap(selectmap<selectmap(id(1)))=0;
    pooledSelect(:,:,featuremap)=selectmap;
    % backward
    net.blobs('pool5').set_diff(pooledSelect);
    net.backward_prefilled();
    re = net.blobs('data').get_diff();
    %crop the actual reception field 
    result=Normimage(re);
    result=uint8(result);
    mask=(re~=0);mask=mask(:,:,1);
    masks1=sum(mask,1);masks2=sum(mask,2);
    a=find(masks1~=0);b=find(masks2~=0);
    if isempty(a)
        x1=1;x2=224;
        y1=1;y2=224;
    else    
        x1=b(1);x2=b(end);
        y1=a(1);y2=a(end);
        de=y2-y1;
    end
    if x2>224
       x2=b(end);x1=x2-de;
    end
    if y2>224
       y2=a(end);y1=y2-de;
    end
    %save the result
    img0 = imresize(img0, [256 256], 'bilinear'); 
    img_crops=img0(17:240,17:240,:);
    im_crop=[img_crops(y1:y2,x1:x2,:) 0.8*result(y1:y2,x1:x2,:)+0.2*img_crops(y1:y2,x1:x2,:)]; 
    imwrite(im_crop,['pic5000_pool5result/result_' num2str(featuremap) '_' num2str(ii) '.jpg']);
end
toc;
end  
%% generate tsne result
clear;clc;
addpath('tSNE_matlab/'); 
for featureid=22:22
    FEA=zeros(512*4*4,50);  % list of "x" in the paper
    fs=cell(250,1); % the corresponding list of deconv pics
    load(['layerpic5000pick_pool5_feature' num2str(featureid) '.mat']); 
    pic=picsort{1}(1:50); % top 50 
    for i=1:50
        load(['ppic5000pick_forpool5/' num2str(pic(i)) '.mat']); % load features we prepared before
        % crop the "x"  
        fea_pad=zeros(size(fea,1)+2,size(fea,2)+2,size(fea,3));
        fea_pad(2:end-1,2:end-1,:)=fea;
        xm=picsort{3}(i,1);
        ym=picsort{3}(i,2);        
        fea0=fea_pad(xm*2-1:xm*2+2,ym*2-1:ym*2+2,:);
        FEA(:,i)=fea0(:);
        % prepare the list of deconv pics path
        fs{i}=['pic5000_pool5result/result_' num2str(featureid) '_' num2str(i) '.jpg'];
    end
train_X=FEA';
no_dims = 2;  
initial_dims = 50;  
perplexity = 30;  
% Run t-SNE  
mappedX = tsne(train_X, [], no_dims, initial_dims, perplexity);  
% load embedding
x= mappedX; 
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));
N = length(fs);
% do a guaranteed quade grid layout by taking nearest neighbor
S1 = 500; % size of final image
S2 = 500;
G = zeros(S1, S2, 3, 'uint8');
s1 = 50; % size of every image thumbnail
s2 = 100;
xnum = S1/s1;
ynum = S2/s2;
used = false(N, 1);

qq=length(1:s1:S1);
abes = zeros(qq*2,2);
i=1;
for a=1:s1:S1
    for b=1:s2:S2
        abes(i,:) = [a,b];
        i=i+1;
    end
end

for i=1:size(abes,1)
    a = abes(i,1);
    b = abes(i,2);
    xf = (a-1)/S1;
    yf = (b-1)/S2;
    dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
    dd(used) = inf; % dont pick these
    [dv,di] = min(dd); % find nearest image

    used(di) = true; % mark as done
    I = imread(fs{di});
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s1, s2]);

    G(a:a+s1-1, b:b+s2-1, :) = I;

    if mod(i,100)==0
        fprintf('%d/%d\n', i, size(abes,1));
    end
end
imshow(G)
imwrite(G,['result_' num2str(featureid) '.jpg']);
end

