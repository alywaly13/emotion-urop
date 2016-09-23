import numpy
import scipy
from scipy import io
from scipy.io import wavfile
import yaafelib
import os
from sklearn import datasets, svm, neighbors, linear_model, cross_validation, grid_search, metrics

def createAFP():
    engine=yaafelib.Engine()
    fp=yaafelib.FeaturePlan(sample_rate=16000)
    fp.addFeature('energy: Energy')
    fp.addFeature('mfcc: MFCC blockSize=2048 stepSize=1024')
    df=fp.getDataFlow()
    engine.load(fp.getDataFlow())
    afp=yaafelib.AudioFileProcessor()
    return afp, engine

def extractFeatures(wavFile, afp, engine):

    afp.processFile(engine, wavFile)
    feats=engine.readAllOutputs()
    mfccarr=feats['mfcc'] 
    enerarr=feats['energy'] 
    allFeats=[]
    for mfccCoeff in mfccarr.T:
        avg=sum(mfccCoeff)/len(mfccCoeff)
        # allFeats=numpy.append(allFeats, avg)
        allFeats.append(avg)
    enerSum=0
    for ener in enerarr.T:
            enerSum+=ener[0]
    allFeats.append(enerSum/len(enerarr))
    return allFeats

def extractFeaturesFiles(waveFiles):
    dataSet=[]
    afp, engine=createAFP()
    for wavfile in waveFiles:
        allFeats=extractFeatures(wavfile, afp, engine)
        dataSet.append(allFeats)
    return dataSet

def extractEmoLabels(wavFiles):
    targetLabels=[]
    for wavfile in wavFiles:
        targetLabels.append(wavfile[-6])
    return numpy.array(targetLabels)
        
def train(audioFilesToTrainOn):
    dataSet=extractFeaturesFiles(audioFilesToTrainOn)
    targetLabels=extractEmoLabels(audioFilesToTrainOn)
    knn = neighbors.KNeighborsClassifier()
    logistic = linear_model.LogisticRegression()
    return logistic.fit(dataSet,targetLabels)

def getLabel(wavFile, model):
    afp, engine=createAFP()
    feats=extractFeatures(wavFile, afp, engine)
    return getEmo(model.predict(feats))

def getEmo(letter):
    if letter=='W':
        return "anger"
    if letter=='A':
        return "angst"
    if letter=='L':
        return "boredom"
    if letter=='E':
        return "disgust"
    if letter=='F':
        return "happiness"
    if letter=='T':
        return "sadness"
    if letter=='N':
        return "neutral"
    else:
        return letter

def test():
    audioFileNames=os.listdir("wav")
    os.chdir("wav")
    n_samples = len(audioFileNames)
    index=int(.9*n_samples)
    model=train(audioFileNames[:index])
    test_data=extractFeaturesFiles(audioFileNames[index:])
    test_labels=extractEmoLabels(audioFileNames[index:])
    #print('Score: %f' % model.score(X_test, y_test))
    return model.score(test_data,test_labels)

def crossValidate(model, x, y, ksplits=3):
    myScorer=metrics.make_scorer(custom_scorer, greater_is_better=True)
    scores = cross_validation.cross_val_score(model, x,y, cv=ksplits, scoring=myScorer)
    return scores

def custom_scorer(target, predictions):
    correct=0
    samecat=0
    weight=0.5
    n=len(predictions)
    categories={"N":"neutral" ,"A":"negative", "W":"negative", "L":"neutral", "E":"negative", "F":"positive", "T":"negative"}
    for i in range(0, n):
        if target[i]==predictions[i]:
            correct+=1
        elif categories[target[i]]==categories[predictions[i]]:
            samecat+=1
    samecat=correct+samecat
    percentCorrect=correct/(float(n))
    percentSameCat=samecat/(float(n))
    return weight*percentCorrect + (1-weight)*percentSameCat

def tuneParams(x,y, model, parameters):
    myScorer=metrics.make_scorer(custom_scorer, greater_is_better=True)
    clf=grid_search.GridSearchCV(model, parameters, scoring=myScorer)
    clf.fit(x,y)
    for item in clf.grid_scores_:
        print item
        
afn=os.listdir("wav")
os.chdir("wav")
x=extractFeaturesFiles(afn)
y=extractEmoLabels(afn)
os.chdir("../")
logisticParams={'penalty':('l1', 'l2'), 'C':[1,5,10]}
logisticModel=linear_model.LogisticRegression()
linSvc1=svm.LinearSVC(penalty='l2', loss='l2', dual=False)
linSvc2=svm.LinearSVC(penalty='l1', loss='l2', dual=False)
linSvc3=svm.LinearSVC(penalty='l2', loss='l2', dual=True)
linSvc4=svm.LinearSVC(penalty='l2', loss='l1', dual=True)
linearSVCParams={'C':[1,2,5]}
knn = neighbors.KNeighborsClassifier()
knnParams={'weights':('uniform', 'distance'), 'algorithm':('ball_tree', 'kd_tree', 'brute')}
svc=svm.SVC()
svcParams={'C':[1.0, 5.0], 'kernel':('rbf', 'poly', 'sigmoid')}
sgd=linear_model.SGDClassifier()
sgdParams={'loss':('hinge', 'log', 'modified_huber'), 'penalty':('l1', 'l2', 'elasticnet')}

allModels=[logisticModel, linSvc1, linSvc2, linSvc3, linSvc4, knn, svc, sgd]
allParams=[logisticParams, linearSVCParams, linearSVCParams, linearSVCParams, linearSVCParams, knnParams, svcParams, sgdParams]

for i in range(0, len(allModels)):
    print allModels[i]
    tuneParams(x,y,allModels[i], allParams[i])
    print "\n"
