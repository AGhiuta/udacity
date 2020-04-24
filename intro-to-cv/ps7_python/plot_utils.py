import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_knn_confusion_matrix(*args):
    if len(args) == 3:
        X_train, y_train, fname, title = args
        classifier = cv2.ml.KNearest_create()
        confusion_matrix = np.zeros((3, 3))

        for i in range(len(X_train)):
            X_test = np.array([X_train[i]])
            y_test = np.array([y_train[i]])

            classifier.train(np.delete(X_train, i, axis=0),
                             cv2.ml.ROW_SAMPLE, np.delete(y_train, i, axis=0))

            _, res, _, _ = classifier.findNearest(X_test, 1)
            confusion_matrix[y_test-1, int(res[0])-1] += 1

        np.set_printoptions(precision=2)
        plot_confusion_matrix(confusion_matrix,
                              classes=["a1", "a2", "a3"],
                              normalize=True,
                              title=title,
                              fname=fname)

        return confusion_matrix

    X_train, y_train, X_test, y_test, fname, title = args

    return __plot_knn_confusion_matrix(
        X_train, y_train, X_test, y_test, fname, title)


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title="Confusion Matrix",
                          fname="conf_mat.png",
                          cmap="Blues"):
    """
    function borrowed from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.figure()
    plt.imshow(confusion_matrix, interpolation="nearest",
               cmap=plt.get_cmap(cmap))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        confusion_matrix = (confusion_matrix * 100 /
                            confusion_matrix.sum()).astype(np.uint) / 100.0

    thresh = confusion_matrix.max() / 2.

    for i, j in itertools.product(range(confusion_matrix.shape[0]),
                                  range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(fname)


def __plot_knn_confusion_matrix(X_train, y_train, X_test, y_test, fname, title):
    classifier = cv2.ml.KNearest_create()
    confusion_matrix = np.zeros((3, 3))

    classifier.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    _, pred, _, _ = classifier.findNearest(X_test, 1)

    for (y, y_pred) in zip(y_test, pred):
        confusion_matrix[y-1, int(y_pred)-1] += 1

    np.set_printoptions(precision=2)
    plot_confusion_matrix(confusion_matrix,
                          classes=["a1", "a2", "a3"],
                          normalize=True,
                          title=title,
                          fname=fname)

    return confusion_matrix
