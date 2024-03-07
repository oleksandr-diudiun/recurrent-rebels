#[macro_use]
use lazy_static::lazy_static;
use ndarray::{Array2};
use serde_derive::Deserialize;
use std::sync::Arc;

const WEIGHTS: &str = r#"{
        
    "W1": [
        [
            0.9772368708434334,
            0.26942003090851785,
            0.01993951099869208,
            0.7259113659590792,
            -0.0828135162540447
        ],
        [
            1.1918258262337948,
            -0.7073274480012282,
            0.4454732214480552,
            0.7421155309768681,
            0.06377224875649294
        ],
        [
            -1.0256981665673595,
            1.4337747372324197,
            0.44081951565405536,
            -0.13619925572550606,
            1.0417113291793343
        ],
        [
            -1.4383727345972976,
            1.1980794694737746,
            0.509709640961367,
            -0.564531359970492,
            1.5255046325726431
        ]
    ],
    "b1": [
        [
            2.00847188680053,
            -1.1155749946012323,
            -0.322647743171443,
            1.2783190402510796,
            -1.456759996416448
        ]
    ],
    "W2": [
        [
            2.1309714713615846,
            0.9432979677913625,
            -2.0307780352432916
        ],
        [
            -1.0075705916925504,
            0.5668664168361047,
            2.0483119637601703
        ],
        [
            0.17023940839262244,
            0.3646353017452193,
            0.7714480792961698
        ],
        [
            0.8685033603895828,
            0.9994249085078235,
            -1.0434088805886796
        ],
        [
            -0.6546825825759535,
            0.15131913458868082,
            2.482932611300431
        ]
    ],
    "b2": [
        [
            0.1837537201470088,
            0.9188784541248332,
            -1.1026321742718477
        ]
    ]

}"#;

lazy_static! {
    // This will hold the converted weights and biases, and will be initialized only once
    static ref W_AND_B_NDARRAY: Arc<WeightsAndBiasesNdarray> = {
        // Normally, you would load the weights string from somewhere. For demonstration, let's assume WEIGHTS is a valid JSON string.
        let w_and_b: WeightsAndBiases = serde_json::from_str(&WEIGHTS).expect("JSON was not well-formatted");
        Arc::new(WeightsAndBiasesNdarray::from(w_and_b))
    };
}

pub fn classify(a: f32, b: f32, c: f32, d: f32) -> i32 {
    let w_and_b_ndarray = Arc::clone(&W_AND_B_NDARRAY);

    // Convert inputs from Rust's Vec to ndarray::Array2
    let inputs = Array2::from_shape_vec((1, 4), vec![a, b, c, d]).unwrap();

    // Forward propagation using the ndarray structures
    let hidden_input = inputs.dot(&w_and_b_ndarray.W1) + &w_and_b_ndarray.b1;
    let hidden_output = hidden_input.map(|a| a.max(0.0)); // ReLU activation function

    let output_input = hidden_output.dot(&w_and_b_ndarray.W2) + &w_and_b_ndarray.b2;
    let output = softmax(output_input);

    find_max_index(&output).unwrap() as i32
}


fn softmax(x: Array2<f32>) -> Array2<f32> {
    // shape=[1]
    let max_per_row = x.map_axis(ndarray::Axis(1), |row| row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    //println!("{:?}", max_per_row);
    let exp = x - &max_per_row.insert_axis(ndarray::Axis(1));
    let exp = exp.mapv(|a| a.exp());
    let sum_exp = exp.sum_axis(ndarray::Axis(1));
    exp / &sum_exp.insert_axis(ndarray::Axis(1))
}



#[derive(Deserialize)]
struct WeightsAndBiases {
    W1: Vec<Vec<f32>>,
    b1: Vec<Vec<f32>>,
    W2: Vec<Vec<f32>>,
    b2: Vec<Vec<f32>>,
}

struct WeightsAndBiasesNdarray {
    W1: Array2<f32>,
    b1: Array2<f32>,
    W2: Array2<f32>,
    b2: Array2<f32>,
}

impl WeightsAndBiasesNdarray {
    // A function to convert WeightsAndBiases into WeightsAndBiasesNdarray
    fn from(w_and_b: WeightsAndBiases) -> Self {
        let weights_ih = Array2::from_shape_vec((4, 5), w_and_b.W1.into_iter().flatten().collect()).unwrap();
        let biases_ih = Array2::from_shape_vec((1, 5), w_and_b.b1.into_iter().flatten().collect()).unwrap();
        let weights_ho = Array2::from_shape_vec((5, 3), w_and_b.W2.into_iter().flatten().collect()).unwrap();
        let biases_ho = Array2::from_shape_vec((1, 3), w_and_b.b2.into_iter().flatten().collect()).unwrap();

        WeightsAndBiasesNdarray {
            W1: weights_ih,
            b1: biases_ih,
            W2: weights_ho,
            b2: biases_ho,
        }
    }
}



fn find_max_index(arr: &Array2<f32>) -> Option<usize> {
    let mut max_value = f32::NEG_INFINITY;
    let mut max_index = None;

    for (index, &value) in arr.iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_index = Some(index);
        }
    }

    max_index
}

