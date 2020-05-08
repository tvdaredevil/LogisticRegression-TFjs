"use strict";
exports.__esModule = true;
var tf = require("@tensorflow/tfjs");
var LogisticRegression_1 = require("./LogisticRegression");
// const strToNumberGrid = v => v.split('|').map(l => l.split('').map(Number))
var strToNumberGrid = function (v) { return tf.tensor(v.replace(/\|/g, '').map(function (l) { return l.split('').map(Number); })); };
function unzip(arr) {
    var table = Array(arr[0].length).fill(0).map(function (_, i) {
        return arr.map(function (e) { return e[i]; });
    });
    return table;
}
//= Data declarations =======================================|
/** The training set */
var train_set = [
    [0, '00100|01010|10001|11111|10001'],
    [0, '11100|10010|11100|10010|11100'],
    [0, '01100|10010|10000|10010|01100'],
    [0, '11100|10010|10010|10010|11100'],
    [0, '11100|10000|11100|10000|11100'],
    [1, '11110|10000|11100|10000|10000'],
    [1, '11110|10000|10000|10111|11110'],
    [1, '10001|10001|11111|10001|10001'],
    [1, '11111|00100|00100|00100|11111'],
    [1, '11110|01000|01000|01010|01110'],
].map(function (_a) {
    var k = _a[0], v = _a[1];
    return [k, strToNumberGrid(v)];
});
/** The test set */
var test_set = [
    '10010|10100|11000|10100|10010',
    '10000|10000|10000|10000|11110',
    '11011|10101|10101|10001|10001',
    '11001|11101|10111|10011|10001',
    '01110|10001|10001|10001|01110',
].map(strToNumberGrid);
//= Training Class =======================================|
var _a = unzip(train_set), _x = _a[0], y = _a[1];
var x = tf.tensor(_x);
var model = new LogisticRegression_1.LogisticRegression([x.shape[0], 1]);
model.forward(x);
