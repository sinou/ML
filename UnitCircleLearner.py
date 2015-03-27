from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy as np


alldata = ClassificationDataSet(2, 1, nb_classes=2)
a = 0.5
b = 0.6
r = 0.4
print(alldata)
for n in xrange(400):
        input = [np.random.ranf(), np.random.ranf()]
        temp = ((input[0] - a)*(input[0] - a)) + ((input[1] - b)*(input[1] - b))
        cla = -1
        if(temp < (r*r)):
			cla = 0
        else:
			cla = 1
        alldata.addSample(input, cla)
		
tstdata, trndata = alldata.splitWithProportion( 0.5 )		
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print("Number of training patterns: ", len(trndata))
print("Input and output dimensions: ", trndata.indim, trndata.outdim)
print("First sample (input, target, class):")
print(trndata['input'][0], trndata['target'][0], trndata['class'][0])


fnn = buildNetwork( trndata.indim, 10, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=2)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

for i in range(20):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    out = out.reshape(X.shape)
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in [0,1]:
        here, _ = where(tstdata['class']==c)
        plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
        contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on
    draw()  # update the plot
ioff()
show()	
