import 'colors'
import * as tf from '@tensorflow/tfjs'
import {LogisticRegression, loss, getAccuracy} from './LogisticRegression'
import {Tensor, Rank} from '@tensorflow/tfjs'

// const strToNumberGrid = v => v.split('|').map(l => l.split('').map(Number))
const strToNumberGrid = (v: string) => v.replace(/\|/g, '').split('').map(Number)
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
    [1, '11110|00100|00100|10100|11100'],
].map(([k, v]) => [k, strToNumberGrid(v as string)] as [number, number[]])

/** The test set */
const test_set = [
    '10010|10100|11000|10100|10010',
    '10000|10000|10000|10000|11110',
    '11011|10101|10101|10001|10001',
    '11001|11101|10111|10011|10001',
    '01110|10001|10001|10001|01110',
].map(strToNumberGrid)

//= Training Class =======================================|
;(_ => {// const [_x, y] = unzip(train_set)
// const x = tf.tensor(_x as number[]) // Flatten x
// const model = new LogisticRegression([x.shape[0], 1])
// const yV = model.forward(x as Tensor<tf.Rank.R0>)

// const cost = loss(yV, y)
// const prediction = predict(yV, y)
// console.log('Cost: ', cost)
// console.log('Acc: ', prediction)

// model.backwards(x as Tensor<tf.Rank.R0>, yV as Tensor<tf.Rank.R0>, y ) 
// model.optimize()
})();

// Full training class
const gr = (t: any, col = 'green') => `${t instanceof Tensor?t.dataSync()[0]:t}`[col]
const costs: number[] = []
const lRate = tf.scalar(1e-8)
const maxIter = 200
const model = new LogisticRegression([train_set[0][1].length, 1], lRate)
console.log(`Created a model of size [${gr(model.gradients.dw.shape.join(', '), 'yellow')}] with learning rate ${gr(lRate.dataSync()[0], 'yellow')}`)

for (let i = 0; i < maxIter; i++) {
    let [_y, _x] = unzip(train_set)
    let [x, y] = [tf.tensor(_x as number[][]), tf.tensor(_y as number[])];
    y = y.expandDims()

    const yV = model.forward(x)
    const cost = loss(yV, y)
    const trainAcc = getAccuracy(yV, y)

    model.backwards(x, yV, y)
    model.optimize()

    if (!(i % 5)) {
        costs.push(cost.dataSync()[0])
        console.log(`=== Cost after iteration ${gr((i+'').padEnd(3))}: ${gr(`${Math.round(costs.slice(-1)[0] * 1e4)/100}%`.padStart(7, ' '), 'red')} | Train Accuracy: ${gr(trainAcc + '%').underline}`)
        console.log('  Labels Expected:', Array.from(y.dataSync()).join(', ').yellow)
        console.log('  Labels Guessed :', Array.from(tf.round(yV).dataSync()).join(', ').yellow)
    }
}
console.log(`Final hyperparams w =`, Array.from(model.gradients.dw.dataSync()), `| b =`, model.gradients.db.dataSync()[0])
const printImageWithLabel = (img, lable) => {
    const chars = [' ', String.fromCharCode(9608)]
    const pr = arr => '  '+arr.map(c => chars[c]).join('')
    console.log(pr(img.slice(0, 5)).green)
    console.log(pr(img.slice(5, 10)).green)
    console.log(pr(img.slice(10, 15)).green,'     -> ', `${lable}`.cyan)
    console.log(pr(img.slice(15, 20)).green)
    console.log(pr(img.slice(20, 25)).green,'\n')
}

console.log('='.repeat(28))
console.log('Predicting on test dataset:')
const yTest = model.forward(tf.tensor(test_set));
const yArr = Array.from(tf.round(yTest).dataSync())
console.log('  Labels Guessed:', yArr.join(', ').yellow)
test_set.forEach((img, i) => printImageWithLabel(img, yArr[i]))