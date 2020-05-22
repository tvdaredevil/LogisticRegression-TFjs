# LogisticRegression-TFjs
A from scratch implementation of Logistic Regression in TensorFlowJs. The weights and biases start at 0 originally
but as training progresses, they will be optimized via gradient descent

In this example, we have a training set of letters that belong to one class or another. After training on those images, we then classify letters as seen in the example below

# To Run
- Download this code to a folder
- Open terminal in that folder (ie. `cd` into this folder)
- Setup all dependencies with `npm i`
  - This will install the project dependencies to `node_modules/`
  - Dependencies like: `tensorflow.js`, `nodemon`, `colors`, etc
- To run the code, please run:
  ```shell
  npm start
  ```

# Example
```shell
$ npm start

Created a model of size [25, 1] with learning rate 9.99999993922529e-9
=== Cost after iteration 0  : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 5  : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 10 : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 15 : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 20 : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 25 : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 30 : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 35 : 0.6931471824645996% | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Cost after iteration 40 : 0.6931471824645996% | Train Accuracy: 60.00000238418579%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
=== Cost after iteration 45 : 0.6931471824645996% | Train Accuracy: 60.00000238418579%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
=== Cost after iteration 50 : 0.6931471824645996% | Train Accuracy: 60.00000238418579%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
=== Cost after iteration 55 : 0.6931471824645996% | Train Accuracy: 60.00000238418579%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
.
.
.
=== Cost after iteration 175: 0.6931470632553101% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 180: 0.6931470632553101% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 185: 0.6931470632553101% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 190: 0.6931470632553101% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 195: 0.6931470632553101% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 200: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 205: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 210: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 215: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 220: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 225: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 230: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 235: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 240: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 245: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 250: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 255: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 260: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 265: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 270: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 275: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 280: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 285: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 290: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 295: 0.6931470036506653% | Train Accuracy: 90.00000357627869%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 0, 1, 1, 1
=== Cost after iteration 300: 0.6931469440460205% | Train Accuracy: 100%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
Final hyperparams w = [
   -0.03999998793005943, -3.5762786065873797e-9,
   0.019999992102384567,   -0.07999997586011887,
  -0.039999984204769135,    0.01999998651444912,
   0.019999993965029716,  -0.039999984204769135,
    0.07999996840953827,  -0.019999993965029716,
   0.039999980479478836,                      0,
  -0.039999984204769135,   -2.38418573772492e-9,
                      0,    0.01999998651444912,
   0.019999993965029716,   -0.03999998793005943,
     0.0599999763071537,  -0.019999997690320015,
  -0.019999993965029716,   0.019999993965029716,
   0.019999993965029716,   -0.03999998793005943,
  -0.019999990239739418
] | b = -3.5762786065873797e-9
============================
Predicting on test dataset:
  Labels Guessed: 1, 0, 1, 1, 0
  █  █ 
  █ █  
  ██         ->  1
  █ █  
  █  █  

  █    
  █    
  █          ->  0
  █    
  ████  

  ██ ██
  █ █ █
  █ █ █      ->  1
  █   █
  █   █ 

  ██  █
  ███ █
  █ ███      ->  1
  █  ██
  █   █ 

   ███ 
  █   █
  █   █      ->  0
  █   █
   ███  
                                                            
   ```
