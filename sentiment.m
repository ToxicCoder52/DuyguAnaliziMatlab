filename = "veri.xlsx";
data = readtable(filename,'TextType','String'); %verileri okuyoruz

cvp = cvpartition(data.Durum,'Holdout',0.2); %train ve test için bölüyoruz
dataTrain = data(cvp.training,:);
dataTest = data(cvp.test,:);

%bolunen kisimlari degiskenlere atiyoruz
textDataTrain = dataTrain.Gorus;
textDataTest = dataTest.Gorus;
YTrain = dataTrain.Durum;
YTest = dataTest.Durum;

documents = tokenizedDocument(textDataTrain);  %verideki gürültü kısmını temizliyoruz
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag,2);
[bag,idx] = removeEmptyDocuments(bag);
YTrain(idx) = [];

%modelin egitilmesi
XTrain = bag.Counts;
mdl = fitcecoc(XTrain,YTrain,'Learners','linear');

%acc hesabi
documentsTest = tokenizedDocument(textDataTest);
XTest = encode(bag,documentsTest);

YPred = predict(mdl,XTest);
acc = sum(YPred == YTest)/numel(YTest);

%%GUI
fig = uifigure;
%text area
txa = uitextarea(fig);
txa.Position = [90 300 400 50];
%acc label
label2 = uilabel(fig,...
    'Position',[240 200 175 15],...
    'Text','');
%analiz butonu
btnSentiment = uibutton(fig,'push',...
                     'Position',[240, 250, 100, 22],...
                     'ButtonPushedFcn', @(btnSentiment,event) AnalyzButtonPushed(btnSentiment,txa,bag,mdl,acc,label2));
btnSentiment.Text = 'Analiz';

%butona basildiginda yapilan islemler
function AnalyzButtonPushed(btnSentiment,txa,bag,mdl,acc,label2)
str=txa.Value;
documentsNew = tokenizedDocument(str);                      
XNew = encode(bag,documentsNew);
labelsNew = predict(mdl,XNew);
label2.Text = "Accuracy = "+num2str(acc);
msgbox(labelsNew);
end

