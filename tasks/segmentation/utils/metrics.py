import logging
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def metrics(predictions, gts, label_values):
    logger = logging.getLogger("dinov3seg")

    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))

    logger.info("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    logger.info("%d pixels processed" % (total))
    logger.info("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        logger.info("%s: %.4f" % (label_values[l_id], score))
    logger.info("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    logger.info("F1Score :")
    for l_id, score in enumerate(F1Score):
        logger.info("%s: %.4f" % (label_values[l_id], score))
    logger.info('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    logger.info("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    logger.info("Kappa: %.4f" % (kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) -
                          np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    logger.info('mean MIoU: %.4f' % (MIoU))
    logger.info("---")

    return MIoU
