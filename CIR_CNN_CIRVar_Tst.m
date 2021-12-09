function ConvNet = CIR_CNN_CIRVar_Tst(X_train,Y_train,X_val,Y_val,feature)
[numBin,numSeries,numChannel,numTrain] = size(X_train);
if feature=="CIR" || feature=="Var"
    lgraph = layerGraph();
    tempLayers = [
        imageInputLayer([numBin,numSeries,numChannel],"Name","imageinput")
        convolution2dLayer([10 1],8,"Name","conv_1","Padding","same")
        batchNormalizationLayer("Name","batchnorm_1")
        reluLayer("Name","relu_1")
        maxPooling2dLayer([10 1],"Name","maxpool_1","Padding","same","Stride",[5 1])
        convolution2dLayer([4 2],16,"Name","conv_2","Padding","same")
        batchNormalizationLayer("Name","batchnorm_2")
        reluLayer("Name","relu_2")
        maxPooling2dLayer([4 2],"Name","maxpool_2","Padding","same","Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([4 1],32,"Name","conv_3","Padding","same","Stride",[2 1])
        batchNormalizationLayer("Name","batchnorm_3")
        reluLayer("Name","relu_3")
        convolution2dLayer([4 1],32,"Name","conv_4","Padding","same")
        batchNormalizationLayer("Name","batchnorm_4")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],32,"Name","conv_5","Padding","same","Stride",[2 1])
        batchNormalizationLayer("Name","batchnorm_5")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_1")
        reluLayer("Name","relu_4")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],64,"Name","conv_8","Padding","same","Stride",[2 1])
        batchNormalizationLayer("Name","batchnorm_8")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([4 1],64,"Name","conv_6","Padding","same","Stride",[2 1])
        batchNormalizationLayer("Name","batchnorm_6")
        reluLayer("Name","relu_5")
        convolution2dLayer([4 1],64,"Name","conv_7","Padding","same")
        batchNormalizationLayer("Name","batchnorm_7")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_2")
        reluLayer("Name","relu_6")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([4 1],128,"Name","conv_9","Padding","same","Stride",[2 1])
        batchNormalizationLayer("Name","batchnorm_9")
        reluLayer("Name","relu_7")
        convolution2dLayer([4 1],128,"Name","conv_10","Padding","same")
        batchNormalizationLayer("Name","batchnorm_10")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],128,"Name","conv_11","Padding","same","Stride",[2 1])
        batchNormalizationLayer("Name","batchnorm_11")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_3")
        reluLayer("Name","relu_8")
        globalMaxPooling2dLayer("Name","gmpool")
        fullyConnectedLayer(10,"Name","fc_1")
        fullyConnectedLayer(1,"Name","fc_2")
        regressionLayer("Name","regressionoutput")];
    lgraph = addLayers(lgraph,tempLayers);
    clear tempLayers;
    lgraph = connectLayers(lgraph,"maxpool_2","conv_3");
    lgraph = connectLayers(lgraph,"maxpool_2","conv_5");
    lgraph = connectLayers(lgraph,"batchnorm_5","addition_1/in2");
    lgraph = connectLayers(lgraph,"batchnorm_4","addition_1/in1");
    lgraph = connectLayers(lgraph,"relu_4","conv_8");
    lgraph = connectLayers(lgraph,"relu_4","conv_6");
    lgraph = connectLayers(lgraph,"batchnorm_8","addition_2/in2");
    lgraph = connectLayers(lgraph,"batchnorm_7","addition_2/in1");
    lgraph = connectLayers(lgraph,"relu_6","conv_9");
    lgraph = connectLayers(lgraph,"relu_6","conv_11");
    lgraph = connectLayers(lgraph,"batchnorm_10","addition_3/in1");
    lgraph = connectLayers(lgraph,"batchnorm_11","addition_3/in2");
else
    
end

%% parameters
miniBatchSize = floor(numTrain/10);
learnRate = 1e-3;
options = trainingOptions('adam', ...
    'InitialLearnRate',learnRate, ...
    'MaxEpochs',50, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{X_val,Y_val}, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress',...
    'Verbose',0);
%% training
ConvNet = trainNetwork(X_train,Y_train,lgraph,options);


end
