% Part 1.

close all; clear all; clc

% Load cropped faces dataset.
m1=192; n1=168; % Resolution of images.
nw1=96*84; % SVD resolutions.
cropped_faces=zeros(32256, 2432); 
% we got 2432 images in croppedYale folder, we are missing yaleB14 folder.

file='/Users/yuhongliu/Downloads/CroppedYale/yaleB';
pos=0;

for i=1:39
    if i<=9
        dirt=append(file,string(0),string(i));
    else
        dirt=append(file,string(i));
    end
    file_ls=dir(dirt);
    NF = length(file_ls);
    for j = 3:NF
        pos=pos+1;
        cropped_faces(:, pos) = reshape(imread(fullfile(dirt, file_ls(j).name)), 32256,1);
    end
end


%%

% Load uncropped faces dataset.
m2=243; n2=320; nw2=122*160;% remember the shape of images.
uncropped_faces=zeros(77760,165); % we have 15 subjects with 11 different faces.

file='/Users/yuhongliu/Downloads/yalefaces_uncropped/yalefaces/subject';
subtitle=[".centerlight",".glasses",".happy",".leftlight",".noglasses",".normal",".rightlight",".sad",".sleepy",".surprised",".wink"];

for i=1:15
    for j=1:11
        if i<=9
            file_title=append(file,string(0),string(i),subtitle(j));
        else
            file_title=append(file,string(i),subtitle(j));
        end
        pos=11*(i-1)+j;
        uncropped_faces(:,pos)=reshape(imread(file_title), 77760, 1);
    end
end

%%

% Get the wavelet representations.
cropped_faces_wave = dc_wavelet(cropped_faces, m1, n1, nw1);
uncropped_faces_wave = dc_wavelet(uncropped_faces, m2, n2, nw2);

feature = 60;

%%

[U1,S1,V1]=svd(cropped_faces_wave,0);
U1=U1(:,1:feature);
[U2,S2,V2]=svd(uncropped_faces_wave,0);
U2=U2(:,1:feature);

% The first four POD modes.
figure(1);
for j = 1:4
    % Cropped faces.
    subplot(4,2,2*j-1);
    ut1=reshape(U1(:,j), 96, 84);
    ut2=ut1(96:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Cropped Eigenfaces with mode' num2str(j)]);
    % Uncropped faces.
    subplot(4,2,2*j);
    ut1=reshape(U2(:,j), 122, 160);
    ut2=ut1(122:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Uncropped Eigenfaces with mode' num2str(j)]);
end

%%

figure(2);
% Cropped faces.
subplot(2,2,1); % normal scale.
plot(diag(S1),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Normal scale: Cropped')
subplot(2,2,3);
semilogy(diag(S1),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Log scale: Cropped')
% Uncropped faces.
subplot(2,2,2); % normal scale.
plot(diag(S2),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Normal scale: Uncropped')
subplot(2,2,4);
semilogy(diag(S2),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Log scale: Uncropped')

%%

% Projection of the first 40 face images onto the first four POD modes.
figure(3);
for j=1:4
    % Cropped faces.
    subplot(4,2,2*j-1);
    plot(1:40,V1(1:40,j),'ko-');
    xlabel('Mode')
    title(['first 40 Cropped images Projection onto mode' num2str(j)])
    % Uncropped faces.
    subplot(4,2,2*j);
    plot(1:40,V2(1:40,j),'ko-');
    xlabel('Mode')
    title(['first 40 Uncropped images Projection onto mode' num2str(j)])
end


%%

function dcData = dc_wavelet(dcfile, m, n, nw)
    [p,q]=size(dcfile);
    nbcol = size(colormap(gray),1);
    dcData=zeros(nw,q);
    
    for i = 1:q
        X=double(reshape(dcfile(:,i),m,n));
        [cA,cH,cV,cD]=dwt2(X,'haar');
        cod_cH1 = wcodemat(cH,nbcol);
        cod_cV1 = wcodemat(cV,nbcol); 
        cod_edge=cod_cH1+cod_cV1; 
        dcData(:,i)=reshape(cod_edge,nw,1);
    end
end

% Test 1. Bands from different genres.

% NEED TO CHANGE GERSHWIN TO ANOTHER BAND FROM JAZZ.
% NEED TO TEST WITH SVM.

close all; clear all; clc

fivesec=240000; % Force all the music clips to be this long.

% Load Vivaldi dataset.

vivaldi=zeros(fivesec, 100); % We have 5 songs and each of them can produce 20 clips.
Fs_v=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test1/Vivaldi';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into classical.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+3).name)); % For some reason, the ls contains three extra things.
    Fs_v(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        vivaldi(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

% Load Gershwin music set.

gershwin=zeros(fivesec, 100); % We have 5 songs and each of them can produce 20 clips.
Fs_g=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test1/Gershwin';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into jazz.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+2).name)); % For some reason, the ls contains three extra things.
    Fs_g(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        gershwin(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

% Load Led Zeppelin music.

zeppelin=zeros(fivesec, 100); % We have 5 songs and each of them can produce 20 clips.
Fs_z=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test1/Zeppelin';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into jazz.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+2).name)); % For some reason, the ls contains two extra '.', '..'.
    Fs_z(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        zeppelin(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

%%

% Samples of Amplitude of vivaldi, gershwin, zeppelin music.
figure(1)
for i=1:5
    subplot(5,3,3*(i-1)+1);
    plot((1:fivesec)/Fs_v(1,i),vivaldi(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end
for i=1:5
    subplot(5,3,3*(i-1)+2);
    plot((1:fivesec)/Fs_g(1,i),gershwin(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end
for i=1:5
    subplot(5,3,3*(i-1)+3);
    plot((1:fivesec)/Fs_z(1,i),zeppelin(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end

%%

% ########################################################################

% Using SVD on the music itself.

X=[vivaldi gershwin zeppelin];

[U1,S1,V1]=svd(X,0);

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S1),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14]);
xlabel('mode'), ylabel('S: Variance explained')
subplot(2,1,2);
semilogy(diag(S1),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14]);
xlabel('mode'), ylabel('S: Variance explained')

% First 4 Mods.
figure(3);
for j = 1:4
    subplot(4,1,j);
    plot(U1(:,j));
    xlabel('time'), ylabel('Amplitude')
end

figure(4)
plot3(V1(1:100,1), V1(1:100, 2), V1(1:100, 3), 'ko') % Vivaldi.
hold on
plot3(V1(101:200,1), V1(101:200, 2), V1(101:200, 3), 'ro') % Gershwin.
hold on
plot3(V1(201:300,1), V1(201:300, 2), V1(201:300, 3), 'go') % Zeppelin.

%%

feature=50;
avg_correct=0;
cross=300;

for i = 1:cross
    
    % figure(4+i)
    
    % Seperate training and testing dataset.
    q1=randperm(100); q2=randperm(100); q3=randperm(100);
    xvivaldi=V1(1:100,1:feature); xgershwin=V1(101:200,1:feature); xzeppelin=V1(201:300,1:feature);
    xtrain=[xvivaldi(q1(1:80),:); xgershwin(q2(1:80),:); xzeppelin(q3(1:80),:)];
    xtest=[xvivaldi(q1(81:end),:); xgershwin(q2(81:end),:); xzeppelin(q3(81:end),:)];

    % GM does not work.
    % gm=fitgmdist(xtrain,3);
    % pre=cluster(gm, xtest);

    ctrain=[ones(80, 1); 2*ones(80,1); 3*ones(80,1)];
    % Naive Bayesian method.
    % This gives the best result. 
    % ########## 0.6373.########
    nb=fitcnb(xtrain, ctrain);
    pre=nb.predict(xtest);
    
    % Linear Discrimination does not behave that well as Naive Bayesian.
    % pre=classify(xtest, xtrain, ctrain);
    
    % SVM.
    % svm=fitcecoc(xtrain,ctrain);
    % pre=predict(svm, xtest);
    
    % bar(pre);
    correct=0;

    for j=1:20
        if pre(j,1)==1
            correct=correct+1;
        end
    end
    for j=21:40
        if pre(j,1)==2
            correct=correct+1;
        end  
    end
    for j=41:60
        if pre(j,1)==3
            correct=correct+1;
        end
    end
    avg_correct = avg_correct + correct/60;
end

disp(avg_correct/cross);

%%

% ########################################################################

% Using SVD on the music spectrogram.

Fs=44100;
[ks_v, vgt_v]=AllSpectrograms(vivaldi, Fs);
[ks_g, vgt_g]=AllSpectrograms(gershwin, Fs);
[ks_z, vgt_z]=AllSpectrograms(zeppelin, Fs);

%%

X=[vgt_v vgt_g vgt_z];

%%
close all; clear all; clc

X = load('spectrogram.mat');
X = X.X;

%%
[U1,S1,V1]=svd(X,0);

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S1),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
subplot(2,1,2);
semilogy(diag(S1),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);

% First 4 Mods.
figure(3);
for j = 1:4
    subplot(4,1,j);
    plot(U1(:,j));
end

figure(4)
plot3(V1(1:100,1), V1(1:100, 2), V1(1:100, 3), 'ko') % Vivaldi.
hold on
plot3(V1(101:200,1), V1(101:200, 2), V1(101:200, 3), 'ro') % Gershwin.
hold on
plot3(V1(201:300,1), V1(201:300, 2), V1(201:300, 3), 'go') % Zeppelin.

%%

feature=40;
avg_correct=0;
cross=300;

for i = 1:cross
    
    % figure(4+i)
    
    % Seperate training and testing dataset.
    q1=randperm(100); q2=randperm(100); q3=randperm(100);
    xvivaldi=V1(1:100,1:feature); xgershwin=V1(101:200,1:feature); xzeppelin=V1(201:300,1:feature);
    xtrain=[xvivaldi(q1(1:80),:); xgershwin(q2(1:80),:); xzeppelin(q3(1:80),:)];
    xtest=[xvivaldi(q1(81:end),:); xgershwin(q2(81:end),:); xzeppelin(q3(81:end),:)];

    % GM does not work.
    % gm=fitgmdist(xtrain,3);
    % pre=cluster(gm, xtest);

    ctrain=[ones(80, 1); 2*ones(80,1); 3*ones(80,1)];
    % Naive Bayesian method.
    % nb=fitcnb(xtrain, ctrain);
    % pre=nb.predict(xtest);
    
    % Linear Discrimination behave the best with spectrogram.
    % 84.00%.
    pre=classify(xtest, xtrain, ctrain);
    
    % SVM.
    % svm=fitcecoc(xtrain,ctrain);
    % pre=predict(svm, xtest);
    
    % bar(pre);
    correct=0;

    for j=1:20
        if pre(j,1)==1
            correct=correct+1;
        end
    end
    for j=21:40
        if pre(j,1)==2
            correct=correct+1;
        end  
    end
    for j=41:60
        if pre(j,1)==3
            correct=correct+1;
        end
    end
    avg_correct = avg_correct + correct/60;
end

disp(avg_correct/cross);

%%

function [dcks,dcdata]=AllSpectrograms(dcfile, Fs)
    [p,q]=size(dcfile);
    dcdata=zeros(2640000,q);
    dcks=zeros(240000,q);
    
    for i=1:q
        [ks, vgt]=spectrogram(dcfile(:,i)', Fs);
        dcks(:,i)=ks'; % Transpose ks from 1x240000 to 240000x1.
        dcdata(:,i)=reshape(vgt, 2640000, 1); % Reshape vgt from 11x240000 to a column vector.
    end
end


%%

function [ks,vgt_spec]=spectrogram(v, Fs)
    n=length(v);
    L=n/Fs;
    t2=linspace(0,L,n+1);
    t=t2(1:n);
    k=(2*pi/L)*[0:n/2-1 -n/2:-1];
    ks=fftshift(k);
    
    vgt_spec=[];
    tslide=0:.5:L;
    for j=1:length(tslide)
        g=exp(-5.0*(t-tslide(j)).^2);
        vg=g.*v; vgt=fft(vg);
        vgt_spec=[vgt_spec; abs(fftshift(vgt))];
        % subplot(3,1,1),plot(t,v,'k',t,g,'r');
        % subplot(3,1,2),plot(t,vg,'k');
        % subplot(3,1,3),plot(ks,abs(fftshift(vgt))/max(abs(vgt)));
        % drawnow;
    end

    % if d == 1
        % pcolor(tslide, ks, vgt_spec.'); shading interp;
        % set(gca,'Ylim',[-10000 10000],'Fontsize',[14]);
        % colormap(hot);
    % end
end

% Test 2. 90s Seattle grunge bands.

% NEED TO CHECK IF THE DATASET IS FINE.
% NEED TO TRY ON SVM.

close all; clear all; clc

fivesec=240000; % Force all the music clips to be this long.

% Load Alice in Chains dataset.

alice=zeros(fivesec, 100); % We have 5 songs and each of them can produce 20 clips.
Fs_a=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test2/AliceInChains';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into classical.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+3).name)); % For some reason, the ls contains three extra things.
    Fs_a(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        alice(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

% Load Pearl Jam music set.

pearl=zeros(fivesec, 100); % We have 5 songs and each of them can produce 20 clips.
Fs_p=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test2/PearlJam';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into jazz.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+3).name)); % For some reason, the ls contains three extra things.
    Fs_p(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        pearl(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

% Load Nirvana music.

nirvana=zeros(fivesec, 100); % We have 5 songs and each of them can produce 20 clips.
Fs_n=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test2/Nirvana';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into jazz.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+3).name)); % For some reason, the ls contains two extra '.', '..'.
    Fs_n(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        nirvana(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

%%

% Samples of Amplitude.
figure(1)
for i=1:5
    subplot(5,3,3*(i-1)+1);
    plot((1:fivesec)/Fs_a(1,i),alice(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end
for i=1:5
    subplot(5,3,3*(i-1)+2);
    plot((1:fivesec)/Fs_p(1,i),pearl(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end
for i=1:5
    subplot(5,3,3*(i-1)+3);
    plot((1:fivesec)/Fs_n(1,i),nirvana(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end

%%

% ########################################################################

% Using SVD on the music itself.

X=[alice pearl nirvana];

[U1,S1,V1]=svd(X,0);

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S1),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14]);
subplot(2,1,2);
semilogy(diag(S1),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14]);

% First 4 Mods.
figure(3);
for j = 1:4
    subplot(4,1,j);
    plot(U1(:,j));
end

figure(4)
plot3(V1(1:100,1), V1(1:100, 2), V1(1:100, 3), 'ko') % Alice In Chains.
hold on
plot3(V1(101:200,1), V1(101:200, 2), V1(101:200, 3), 'ro') % Pearl Jam.
hold on
plot3(V1(201:300,1), V1(201:300, 2), V1(201:300, 3), 'go') % Nirvana.

%%

feature=50;
avg_correct=0;
cross=300;

for i = 1:cross
    
    % figure(4+i)
    
    % Seperate training and testing dataset.
    q1=randperm(100); q2=randperm(100); q3=randperm(100);
    xvivaldi=V1(1:100,1:feature); xgershwin=V1(101:200,1:feature); xzeppelin=V1(201:300,1:feature);
    xtrain=[xvivaldi(q1(1:80),:); xgershwin(q2(1:80),:); xzeppelin(q3(1:80),:)];
    xtest=[xvivaldi(q1(81:end),:); xgershwin(q2(81:end),:); xzeppelin(q3(81:end),:)];

    % GM does not work.
    % gm=fitgmdist(xtrain,3);
    % pre=cluster(gm, xtest);

    ctrain=[ones(80, 1); 2*ones(80,1); 3*ones(80,1)];
    % Naive Bayesian method.
    % This gives the best result: 61.07%.
    nb=fitcnb(xtrain, ctrain);
    pre=nb.predict(xtest);
    
    % Linear Discrimination does not behave that well as Naive Bayesian.
    % 45.96%.
    % pre=classify(xtest, xtrain, ctrain);
    
    % SVM. 33.71%.
    % svm=fitcecoc(xtrain,ctrain);
    % pre=predict(svm, xtest);
    
    % bar(pre);
    correct=0;

    for j=1:20
        if pre(j,1)==1
            correct=correct+1;
        end
    end
    for j=21:40
        if pre(j,1)==2
            correct=correct+1;
        end  
    end
    for j=41:60
        if pre(j,1)==3
            correct=correct+1;
        end
    end
    avg_correct = avg_correct + correct/60;
end

disp(avg_correct/cross);

%%

% ########################################################################

% Using SVD on the music spectrogram.

Fs=44100;
vgt_a=AllSpectrograms(alice, Fs);
vgt_p=AllSpectrograms(pearl, Fs);
vgt_n=AllSpectrograms(nirvana, Fs);

%%

X=[vgt_a vgt_p vgt_n];

close all; clear all; clc

X = load('spectrogram2.mat'); 
X = X.X;

%%
[U1,S1,V1]=svd(X,0);

%%

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S1),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
subplot(2,1,2);
semilogy(diag(S1),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);

% First 4 Mods.
figure(3);
for j = 1:4
    subplot(4,1,j);
    plot(U1(:,j));
end

figure(4)
plot3(V1(1:100,1), V1(1:100, 2), V1(1:100, 3), 'ko') % Alice In Chains.
hold on
plot3(V1(101:200,1), V1(101:200, 2), V1(101:200, 3), 'ro') % Pearl Jam.
hold on
plot3(V1(201:300,1), V1(201:300, 2), V1(201:300, 3), 'go') % Nirvana.

%%

feature=40;
avg_correct=0;
cross=300;

for i = 1:cross
    
    % figure(4+i)
    
    % Seperate training and testing dataset.
    q1=randperm(100); q2=randperm(100); q3=randperm(100);
    xvivaldi=V1(1:100,1:feature); xgershwin=V1(101:200,1:feature); xzeppelin=V1(201:300,1:feature);
    xtrain=[xvivaldi(q1(1:80),:); xgershwin(q2(1:80),:); xzeppelin(q3(1:80),:)];
    xtest=[xvivaldi(q1(81:end),:); xgershwin(q2(81:end),:); xzeppelin(q3(81:end),:)];

    % GM does not work.
    % gm=fitgmdist(xtrain,3);
    % pre=cluster(gm, xtest);

    ctrain=[ones(80, 1); 2*ones(80,1); 3*ones(80,1)];
    % Naive Bayesian method. %78.29.
    % nb=fitcnb(xtrain, ctrain);
    % pre=nb.predict(xtest);
    
    % Linear Discrimination bahaves the best.
    % 91.78%.
    pre=classify(xtest, xtrain, ctrain);
    
    % SVM.
    % svm=fitcecoc(xtrain,ctrain);
    % pre=predict(svm, xtest);
    
    % bar(pre);
    correct=0;

    for j=1:20
        if pre(j,1)==1
            correct=correct+1;
        end
    end
    for j=21:40
        if pre(j,1)==2
            correct=correct+1;
        end  
    end
    for j=41:60
        if pre(j,1)==3
            correct=correct+1;
        end
    end
    avg_correct = avg_correct + correct/60;
end

disp(avg_correct/cross);

%%

function dcdata=AllSpectrograms(dcfile, Fs)
    [p,q]=size(dcfile);
    dcdata=zeros(2640000,q);
    
    for i=1:q
        vgt=spectrogram(dcfile(:,i)', Fs);
        dcdata(:,i)=reshape(vgt, 2640000, 1); % Reshape vgt from 11x240000 to a column vector.
    end
end


%%

function vgt_spec=spectrogram(v, Fs)
    n=length(v);
    L=n/Fs;
    t2=linspace(0,L,n+1);
    t=t2(1:n);
    k=(2*pi/L)*[0:n/2-1 -n/2:-1];
    ks=fftshift(k);
    
    vgt_spec=[];
    tslide=0:.5:L;
    for j=1:length(tslide)
        g=exp(-5.0*(t-tslide(j)).^2);
        vg=g.*v; vgt=fft(vg);
        vgt_spec=[vgt_spec; abs(fftshift(vgt))];
        % subplot(3,1,1),plot(t,v,'k',t,g,'r');
        % subplot(3,1,2),plot(t,vg,'k');
        % subplot(3,1,3),plot(ks,abs(fftshift(vgt))/max(abs(vgt)));
        % drawnow;
    end

    % if d == 1
        % pcolor(tslide, ks, vgt_spec.'); shading interp;
        % set(gca,'Ylim',[-10000 10000],'Fontsize',[14]);
        % colormap(hot);
    % end
end

% Test 3. Three genres with different bands.

close all; clear all; clc

fivesec=240000; % Force all the music clips to be this long.

% Load classical dataset.

classical=zeros(fivesec, 100); % We have 5 songs from 5 bands and each of them can produce 20 clips.
Fs_c=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test3/Classical';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into classical.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+3).name)); % For some reason, the ls contains three extra things.
    Fs_c(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        classical(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

% Load Jazz music set.

jazz=zeros(fivesec, 100); % We have 5 songs from 5 bands and each of them can produce 20 clips.
Fs_j=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test3/Jazz';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into jazz.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+3).name)); % For some reason, the ls contains three extra things.
    Fs_j(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        jazz(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

% Load Rock music.

rock=zeros(fivesec, 100); % We have 5 songs and each of them can produce 20 clips.
Fs_r=zeros(1,5); % Sample rates of the 5 songs.
file_path='/Users/yuhongliu/Downloads/Test3/Rock';

file_ls=dir(file_path); % Get all the files info in the folder.
pos=0; % Memorize where to put each clip into jazz.
for i = 1:5 
    [file,Fs_i]=audioread(fullfile(file_path, file_ls(i+3).name)); % For some reason, the ls contains two extra '.', '..'.
    Fs_r(1,i)=Fs_i;
    file=file(:,1);
    for j=1:20 % We need 20 clips.
        pos=pos+1;
        rock(:,pos)=file((j-1)*fivesec+1:j*fivesec);
    end
end

%%

% Samples of Amplitude.
figure(1)
for i=1:5
    subplot(5,3,3*(i-1)+1);
    plot((1:fivesec)/Fs_c(1,i),classical(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end
for i=1:5
    subplot(5,3,3*(i-1)+2);
    plot((1:fivesec)/Fs_j(1,i),jazz(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end
for i=1:5
    subplot(5,3,3*(i-1)+3);
    plot((1:fivesec)/Fs_r(1,i),rock(:,(i-1)*20+5));
    xlabel('Time [sec]'); ylabel('Amplitude');
end

%%

% ########################################################################

% Using SVD on the music itself.

X=[classical jazz rock];

[U1,S1,V1]=svd(X,0);

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S1),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14]);
subplot(2,1,2);
semilogy(diag(S1),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14]);

% First 4 Mods.
figure(3);
for j = 1:4
    subplot(4,1,j);
    plot(U1(:,j));
end

figure(4)
plot3(V1(1:100,1), V1(1:100, 2), V1(1:100, 3), 'ko') % Vivaldi.
hold on
plot3(V1(101:200,1), V1(101:200, 2), V1(101:200, 3), 'ro') % Gershwin.
hold on
plot3(V1(201:300,1), V1(201:300, 2), V1(201:300, 3), 'go') % Zeppelin.

%%

feature=60;
avg_correct=0;
cross=300;

for i = 1:cross
    
    % figure(4+i)
    
    % Seperate training and testing dataset.
    q1=randperm(100); q2=randperm(100); q3=randperm(100);
    xvivaldi=V1(1:100,1:feature); xgershwin=V1(101:200,1:feature); xzeppelin=V1(201:300,1:feature);
    xtrain=[xvivaldi(q1(1:80),:); xgershwin(q2(1:80),:); xzeppelin(q3(1:80),:)];
    xtest=[xvivaldi(q1(81:end),:); xgershwin(q2(81:end),:); xzeppelin(q3(81:end),:)];

    % GM does not work.
    % gm=fitgmdist(xtrain,3);
    % pre=cluster(gm, xtest);

    ctrain=[ones(80, 1); 2*ones(80,1); 3*ones(80,1)];
    % Naive Bayesian method.
    % This gives the best result: 56.19%.
    nb=fitcnb(xtrain, ctrain);
    pre=nb.predict(xtest);
    
    % Linear Discrimination does not behave that well as Naive Bayesian.
    % pre=classify(xtest, xtrain, ctrain);
    
    % SVM.
    % svm=fitcsvm(xtrain, ctrain);
    % pre=ClassificationSVM(svm, xtest);
    % This behaves the worst. 33.62%.
    % svm=fitcecoc(xtrain,ctrain);
    % pre=predict(svm, xtest);
    
    % bar(pre);
    correct=0;

    for j=1:20
        if pre(j,1)==1
            correct=correct+1;
        end
    end
    for j=21:40
        if pre(j,1)==2
            correct=correct+1;
        end  
    end
    for j=41:60
        if pre(j,1)==3
            correct=correct+1;
        end
    end
    avg_correct = avg_correct + correct/60;
end

disp(avg_correct/cross);

%%

% ########################################################################

% Using SVD on the music spectrogram.

Fs=44100;
vgt_c=AllSpectrograms(classical, Fs);
vgt_j=AllSpectrograms(jazz, Fs);
vgt_r=AllSpectrograms(rock, Fs);

%%

X=[vgt_c vgt_j vgt_r];

%%

close all; clear all; clc

X = load('spectrogram3.mat'); 
X = X.X;

%%

[U1,S1,V1]=svd(X,0);

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S1),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
subplot(2,1,2);
semilogy(diag(S1),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);

% First 4 Mods.
figure(3);
for j = 1:4
    subplot(4,1,j);
    plot(U1(:,j));
end

figure(4)
plot3(V1(1:100,1), V1(1:100, 2), V1(1:100, 3), 'ko') % Classical.
hold on
plot3(V1(101:200,1), V1(101:200, 2), V1(101:200, 3), 'ro') % Jazz.
hold on
plot3(V1(201:300,1), V1(201:300, 2), V1(201:300, 3), 'go') % Rock.

%%

feature=40;
avg_correct=0;
cross=300;

for i = 1:cross
    
    % figure(4+i)
    
    % Seperate training and testing dataset.
    q1=randperm(100); q2=randperm(100); q3=randperm(100);
    xvivaldi=V1(1:100,1:feature); xgershwin=V1(101:200,1:feature); xzeppelin=V1(201:300,1:feature);
    xtrain=[xvivaldi(q1(1:80),:); xgershwin(q2(1:80),:); xzeppelin(q3(1:80),:)];
    xtest=[xvivaldi(q1(81:end),:); xgershwin(q2(81:end),:); xzeppelin(q3(81:end),:)];

    % GM does not work.
    % gm=fitgmdist(xtrain,3);
    % pre=cluster(gm, xtest);

    ctrain=[ones(80, 1); 2*ones(80,1); 3*ones(80,1)];
    % Naive Bayesian method. 61.59%.
    % nb=fitcnb(xtrain, ctrain);
    % pre=nb.predict(xtest);
    
    % Linear Discrimination bahaves the best.
    % 75.88%.
    pre=classify(xtest, xtrain, ctrain);
    
    % SVM.
    % 73.25%.
    % svm=fitcecoc(xtrain,ctrain);
    % pre=predict(svm, xtest);
    
    % bar(pre);
    correct=0;

    for j=1:20
        if pre(j,1)==1
            correct=correct+1;
        end
    end
    for j=21:40
        if pre(j,1)==2
            correct=correct+1;
        end  
    end
    for j=41:60
        if pre(j,1)==3
            correct=correct+1;
        end
    end
    avg_correct = avg_correct + correct/60;
end

disp(avg_correct/cross);

%%

function dcdata=AllSpectrograms(dcfile, Fs)
    [p,q]=size(dcfile);
    dcdata=zeros(2640000,q);
    
    for i=1:q
        vgt=spectrogram(dcfile(:,i)', Fs);
        dcdata(:,i)=reshape(vgt, 2640000, 1); % Reshape vgt from 11x240000 to a column vector.
    end
end


%%

function vgt_spec=spectrogram(v, Fs)
    n=length(v);
    L=n/Fs;
    t2=linspace(0,L,n+1);
    t=t2(1:n);
    k=(2*pi/L)*[0:n/2-1 -n/2:-1];
    ks=fftshift(k);
    
    vgt_spec=[];
    tslide=0:.5:L;
    for j=1:length(tslide)
        g=exp(-5.0*(t-tslide(j)).^2);
        vg=g.*v; vgt=fft(vg);
        vgt_spec=[vgt_spec; abs(fftshift(vgt))];
        % subplot(3,1,1),plot(t,v,'k',t,g,'r');
        % subplot(3,1,2),plot(t,vg,'k');
        % subplot(3,1,3),plot(ks,abs(fftshift(vgt))/max(abs(vgt)));
        % drawnow;
    end

    % if d == 1
        % pcolor(tslide, ks, vgt_spec.'); shading interp;
        % set(gca,'Ylim',[-10000 10000],'Fontsize',[14]);
        % colormap(hot);
    % end
end
