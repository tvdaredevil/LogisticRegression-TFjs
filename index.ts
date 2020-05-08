import * as tf from '@tensorflow/tfjs'
import {LogisticRegression} from './LogisticRegression'

// const strToNumberGrid = v => v.split('|').map(l => l.split('').map(Number))
const strToNumberGrid = v => tf.tensor(v.replace(/\|/g, '').map(l => l.split('').map(Number)))
function unzip<T>(arr: T[][]) {
    const table = Array(arr[0].length).fill(0).map((_, i) => {
        return arr.map(e => e[i])
    })
    return table
}

//= Data declarations =======================================|

/** The training set */
const train_set = [
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
].map(([k,v]) => [k, strToNumberGrid(v)])

/** The test set */
const test_set = [
    '10010|10100|11000|10100|10010',
    '10000|10000|10000|10000|11110',
    '11011|10101|10101|10001|10001',
    '11001|11101|10111|10011|10001',
    '01110|10001|10001|10001|01110',
].map(strToNumberGrid)

//= Training Class =======================================|
const [_x, y] = unzip(train_set)
const x = tf.tensor(_x as number[])
const model = new LogisticRegression([x.shape[0], 1])
model.forward(x as tf.Tensor<tf.Rank.R0>)

