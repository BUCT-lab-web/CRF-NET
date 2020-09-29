def confusionMatrix(label, pred, ):
    """
    result: [TP, FN, FP, TN]
    actual_gt: [0, 1, ..., class_num=9]
    """
    matrix = confusion_matrix(label, pred)
    result = np.zeros((class_number, 4))
    for i in range(matrix.shape[0]):
        result[i][0] = matrix.diagonal()[i]
        result[i][1] = matrix[i].sum() - matrix.diagonal()[i]
        result[i][2] = matrix[:, i].sum() - matrix.diagonal()[i]
        result[i][3] = matrix.sum() - matrix[i].sum() - matrix[:, i].sum() + matrix.diagonal()[i]
    return result


def aa(result):
    acc = 0
    for i in range(result.shape[0]):
        acc += result[i][0] / (result[i][0] + result[i][2])
    print("map:{:.5f}".format(acc))


def kappa(result):
    a = 0
    num = result.sum()
    for i in range(result.shape[0]):
        a += (result[i][0] + result[i][2])*(result[i][0] + result[i][1])
    kappa = a/num/num
    print('kappa:{:.5f}'.format(kappa))