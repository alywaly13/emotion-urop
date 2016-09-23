import numpy
import scipy
from scipy import io
from scipy.io import wavfile
import yaafelib2 as yaafelib
import os
from sklearn import datasets, ensemble, svm, neighbors, \
     linear_model, cross_validation, preprocessing, grid_search, metrics

def createAFP():
    engine=yaafelib.Engine()
    fp=yaafelib.FeaturePlan(sample_rate=44100, resample=True, normalize=0.1)
    fp.addFeature('mfcc: MFCC blockSize=2048 stepSize=1024')
    fp.addFeature('autocorr: AutoCorrelation ACNbCoeffs=20 blockSize=2048 stepSize=1024')
    fp.addFeature('mel: MelSpectrum blockSize=2048 stepSize=1024')
    fp.addFeature('obsi: OBSI blockSize=2048 stepSize=1024')

    df=fp.getDataFlow()
    engine.load(fp.getDataFlow())
    afp=yaafelib.AudioFileProcessor()
    return afp, engine

def extractFeatures(wavFile, afp, engine):
    
    afp.processFile(engine, wavFile)
    feats=engine.readAllOutputs()
    allFeats=[]

    for feature in feats:
        featarr=feats[feature]
        for featCoeff in featarr.T:
            avg=sum(featCoeff)/len(featCoeff)
            allFeats.append(avg)
    
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
        if wavfile[0]=="s":
            targetLabels.append(wavfile[0:2])
        elif wavfile[0].isalpha():
            targetLabels.append(wavfile[0])     
        elif wavfile[-6].isalpha():
            targetLabels.append(getEmo(wavfile[-6]))
        else:
            print("unrecognized wavfile name")
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
    weight=0.1
    n=len(predictions)
    categories={"a":"neg", "d":"neg", "f":"neg", "sa":"neg", "ast":"neg",\
                "h":"pos", "n":"n", "su":"n", "b":"n"}
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
        print(item)
##    print clf.best_score_

def main():        
    folders=["wav","KL", "JK", "JE", "DC"]
    x=[]
    y=numpy.array([])
    for folderName in folders:
        afn=os.listdir(folderName)
        os.chdir(folderName)
        x=x+extractFeaturesFiles(afn)
        y=numpy.concatenate([y,extractEmoLabels(afn)])
        os.chdir("../")
    return (x,y)

def main2(x, y):
    x=preprocessing.scale(x)
    logisticParams={'penalty':('l1', 'l2'), 'C':[1,5,10]}
    logisticModel=linear_model.LogisticRegression(tol=0.1)
    linSvc1=svm.LinearSVC(penalty='l2', loss='l2', dual=False)
    linSvc2=svm.LinearSVC(penalty='l1', loss='l2', dual=False)
    linSvc3=svm.LinearSVC(penalty='l2', loss='l2', dual=True)
    linSvc4=svm.LinearSVC(penalty='l2', loss='l1', dual=True)
    linearSVCParams={'C':[1,2,5]}
    knn = neighbors.KNeighborsClassifier()
    knnParams={'weights':('uniform', 'distance'), 'algorithm':('ball_tree', 'kd_tree', 'brute')}
    svc=svm.SVC(max_iter=1000000)
    randomForest=ensemble.RandomForestClassifier()
    randomForestParams={'n_estimators':[5,10,20], 'criterion':('gini', 'entropy')}
    svcParams={'C':[1.0, 5.0], 'kernel':('rbf', 'poly', 'sigmoid')}
    sgd=linear_model.SGDClassifier()
    sgdParams={'loss':('hinge', 'log', 'modified_huber'), 'penalty':('l1', 'l2', 'elasticnet')}
    allModels=[linSvc1, linSvc2, linSvc3, linSvc4]
    allParams=[linearSVCParams, linearSVCParams, linearSVCParams, linearSVCParams]
    allNames=['1', '2', '3', '4']
    linSvc3=svm.LinearSVC(penalty='l2', loss='l2', dual=True, C=1)
   # allModels=[randomForest, logisticModel, linSvc1, linSvc2, linSvc3, linSvc4, knn, svc, sgd]
  #  allParams=[randomForestParams, logisticParams, linearSVCParams, linearSVCParams, linearSVCParams, linearSVCParams, knnParams, svcParams, sgdParams]
  #  allNames=['randomForest', 'logisticModel', 'linSvc1', 'linSvc2', 'linSvc3', 'linSvc4', 'knn', 'svc', 'sgd']
    for i in range(0, len(allModels)):
        print(allNames[i])
        tuneParams(x,y,allModels[i], allParams[i])

