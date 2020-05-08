"use strict";
exports.__esModule = true;
var tf = require("@tensorflow/tfjs");
var mul = tf.mul, add = tf.add, sum = tf.sum, div = tf.div, sub = tf.sub, matMul = tf.matMul, log = tf.log;
/**
 * From-scratch implementation of the Logistic Regression
 */
var LogisticRegression = /** @class */ (function () {
    function LogisticRegression(dim, lRate) {
        if (lRate === void 0) { lRate = tf.scalar(.01); }
        this.lRate = lRate;
        this.w = tf.zeros(dim, 'float32');
        this.b = tf.scalar(0);
        this.gradients = {
            dw: tf.zeros(dim, 'float32'),
            db: tf.scalar(0)
        };
    }
    LogisticRegression.prototype.forward = function (x) {
        var z = matMul(this.w.transpose(), x);
        return this.sigmoid(z);
    };
    LogisticRegression.prototype.sigmoid = function (z) {
        return tf.sigmoid(z);
    };
    /**
     * Updates the weights
     */
    LogisticRegression.prototype.optimize = function () {
        /* w = w - lRate * grads.dw */
        this.w = sub(this.w, mul(this.lRate, this.gradients.dw));
        /* b = b - lRate * grads.db */
        this.b = sub(this.b, mul(this.lRate, this.gradients.db));
    };
    /**
     * Back propogate the updates
     * @param x - The input dataset
     * @param yV - Predicted Labels
     * @param y - Actual labels
     */
    LogisticRegression.prototype.backwards = function (x, yV, y) {
        this.gradients.dw = mul(div(1, x.shape[1]), matMul(x, sub(yV, y).transpose()));
        this.gradients.db = mul(div(1, x.shape[1]), sum(sub(yV, y)));
    };
    return LogisticRegression;
}());
exports.LogisticRegression = LogisticRegression;
/**
 * Calculate the loss between epochs
 * @param yV - Predicted Labels
 * @param y - Actual labels
 */
function loss(yV, y) {
    var m = y.shape[1];
    /* -(1/m) * sum(y * log(yV) + (1-y) * log(1 - yV)) */
    return mul(-(1 / m), sum(add(mul(y, log(yV)), mul(sub(1, y), log(sub(1, yV))))));
}
exports.loss = loss;
/**
 * Predict a label
 * @param yV - Predicted Labels
 * @param y - Actual labels
 */
function predict(yV, y) {
    // Todo(tushar): Look at this line!
    var predict = tf.zeros([1, y.shape[1]]).bufferSync();
    var y_buf = y.bufferSync();
    for (var i = 0; i < yV.shape[1]; i++) {
        predict.set(Math.round(y_buf.get(0, i)), 0, i);
    }
    return sub(100, mul(tf.mean(tf.abs(sub(predict.toTensor(), y))), 100));
}
exports.predict = predict;
