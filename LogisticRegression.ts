import * as tf from '@tensorflow/tfjs'

const {mul, add, sum, div, sub, matMul, log} = tf;

/**
 * From-scratch implementation of the Logistic Regression
 * @param w Represents the weights
 * @param b Represents the biases
 * @param gradients Holds the gradient descent values for weights and biases
 */
export class LogisticRegression {
    w: tf.Scalar
    b: tf.Scalar
    gradients: {dw: tf.Tensor<tf.Rank>; db: tf.Scalar}

    constructor(dim: [number, number], private lRate = tf.scalar(.01)) {
        this.w = tf.zeros(dim, 'float32')
        this.b = tf.scalar(0)
        this.gradients = {
            dw: tf.zeros(dim, 'float32'),
            db: tf.scalar(0),
        }
    }
    /**
     * This method is essentially the "predict" function and is
     * where the model uses its weights to predict an output for a given input
     * @param x Represents any Tensor Data that needs to be labeled by the model
     */
    forward(x: tf.Tensor) {
        const z = matMul(this.w.transpose(), x.transpose())
        return tf.sigmoid(z)
    }

    /**
     * Updates the weights
     */
    optimize() {
        /* w = w - lRate * grads.dw */
        this.w = sub(this.w, mul(this.lRate, this.gradients.dw));

        /* b = b - lRate * grads.db */
        this.b = sub(this.b, mul(this.lRate, this.gradients.db));
    }

    /**
     * Back propogate the updates
     * @param x - The input dataset
     * @param yV - Predicted Labels
     * @param y - Actual labels
     */
    backwards(x: tf.Tensor, yV: tf.Tensor, y: tf.Tensor) {
        this.gradients.dw = mul(div(1, x.shape[1]), matMul(x.transpose(), sub(yV, y).transpose()))
        this.gradients.db = mul(div(1, x.shape[1]), sum(sub(yV, y)))
    }
}

/**
 * Calculate the loss between epochs
 * @param yV - Predicted Labels
 * @param y - Actual labels
 */
export function loss(yV: tf.Tensor, y: tf.Tensor) {
    // console.log('Loss between yhat', yV.shape, 'and y', y.shape)
    const m = y.shape[1]
    /* -(1/m) * sum(y * log(yV) + (1-y) * log(1 - yV)) */
    return mul(-(1/m), sum(
        add(
            mul(y, log(yV)),
            mul(
                sub(1, y),
                log(sub(1, yV)),
            ),
        ),
    ))
}

/**
 * Fetches Accuracy between predicted labels vs actual data
 * @param yV - Predicted Labels
 * @param y - Actual labels
 */
export function getAccuracy(yV: tf.Tensor, y: tf.Tensor) {
    const predictedVals = tf.round(yV)
    return tf.metrics.binaryAccuracy(predictedVals, y).dataSync()[0] * 100
    // sub(100, mul(tf.mean(tf.abs(sub(getAccuracy.toTensor(), y))), 100))
}
