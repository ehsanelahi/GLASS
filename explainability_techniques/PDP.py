import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.inspection as ins
import copy


font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

def partialDependencePlot(model, X_train, features, kind, path=None):
    pdp = ins.PartialDependenceDisplay.from_estimator(model, X_train, features, kind=kind, percentiles=(0, 1))
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()
    return pdp


def multiplyPdps(pdps, path=None):
    res = copy.deepcopy(pdps[0])
    for pdp in pdps[1:]:
        for i in range(len(res.pd_results[0]["individual"][0])):
            res.pd_results[0]["individual"][0][i, :] *= pdp.pd_results[0]["individual"][0][i, :]
        res.pd_results[0]["average"][0, :] *= pdp.pd_results[0]["average"][0, :]
    res.plot()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()
    return res


def getMaximalsOfLine(pdp, lineIndex):
    if lineIndex >= 0:
        yVals = pdp.pd_results[0]["individual"][0][lineIndex, :]
    else:
        yVals = pdp.pd_results[0]["average"][0, :]
    xVals = pdp.pd_results[0]["values"][0]

    maxX = xVals[np.where(yVals == yVals.max())]

    return maxX


def getMaxOfLine(pdp, lineIndex):
    if lineIndex >= 0:
        yVals = pdp.pd_results[0]["individual"][0][lineIndex, :]
    else:
        yVals = pdp.pd_results[0]["average"][0, :]

    return yVals.max()


def getMaximalsOfMeanLine(pdp):
    return getMaximalsOfLine(pdp, -1)


def getMaxOfMeanLine(pdp):
    return getMaxOfLine(pdp, -1)


def getSlope(pdp, x, lineIndex):
    xVals = pdp.pd_results[0]["values"][0]
    linesY = pdp.pd_results[0]["individual"][0][lineIndex, :]
    index = len(xVals) - 1
    for i in range(len(xVals)):
        if xVals[i] > x:
            index = i
            break
    if index == 0:
        index = 1
    slope = (linesY[i] - linesY[index - 1]) / (xVals[index] - xVals[index - 1])
    return slope


def getGlobalMax(pdp, lineIndex=-1):
    """
    Get the (x, y) pair corresponding to the global maximum of a PDP curve.

    Parameters
    ----------
    pdp : PartialDependenceDisplay
        The PDP object returned by partialDependencePlot or multiplyPdps.
    lineIndex : int, optional (default=-1)
        If >= 0, use the individual PDP line for the given neighbor index.
        If -1, use the average PDP curve.

    Returns
    -------
    best_x : float
        Feature value (x-axis) where PDP reaches maximum.
    best_y : float
        Maximum predicted probability at that feature value.
    """
    xVals = pdp.pd_results[0]["values"][0]

    if lineIndex >= 0:
        yVals = pdp.pd_results[0]["individual"][0][lineIndex, :]
    else:
        yVals = pdp.pd_results[0]["average"][0, :]

    max_idx = int(np.argmax(yVals))
    best_x = xVals[max_idx]
    best_y = yVals[max_idx]

    return best_x, best_y



