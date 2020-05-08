import * as tf from '@tensorflow/tfjs'

const {mul, add, sum, div, sub, matMul, log} = tf;

/**
 * From-scratch implementation of the Logistic Regression
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

    forward(x: tf.Scalar) {
        const z = matMul(this.w.transpose(), x)
        return this.sigmoid(z)
    }

    sigmoid(z: tf.Tensor<tf.Rank>) {
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
    backwards(x: tf.Scalar, yV: tf.Scalar, y: tf.Scalar) {
        this.gradients.dw = mul(div(1, x.shape[1]), matMul(x, sub(yV, y).transpose()))
        this.gradients.db = mul(div(1, x.shape[1]), sum(sub(yV, y)))
    }
}

/**
 * Calculate the loss between epochs
 * @param yV - Predicted Labels
 * @param y - Actual labels
 */
export function loss(yV: tf.Scalar, y: tf.Scalar) {
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
 * Predict a label
 * @param yV - Predicted Labels
 * @param y - Actual labels
 */
export function predict(yV: tf.Scalar, y: tf.Scalar) {
    // Todo(tushar): Look at this line!
    const predict = tf.zeros([1, y.shape[1]]).bufferSync()
    const y_buf = y.bufferSync()
    for(let i = 0; i < yV.shape[1]; i++) {
        predict.set(Math.round(y_buf.get(0, i)), 0, i)
    }
    return sub(100, mul(tf.mean(tf.abs(sub(predict.toTensor(), y))), 100))
}
