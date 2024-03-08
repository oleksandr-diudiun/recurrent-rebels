use serde::{Deserialize, Serialize};
use warp::{http::StatusCode, Filter, Rejection, Reply};
// Import the classify function from the classification module.
mod classification;
use classification::classify;

use warp::filters::log::custom;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn, LevelFilter};
use simple_logger::SimpleLogger;

#[tokio::main]
async fn main() {
    

    // Initialize the logger
    let logger = SimpleLogger::new().with_level(LevelFilter::Info).with_utc_timestamps() ;
    match logger.init() {
        Ok(_) => println!("Logger initialized successfully."),
        Err(e) => eprintln!("Logger initialization failed: {}", e),
    }
    
    

    let log = Arc::new(Mutex::new(())); // Create a shared, thread-safe lock for logging


    let sum_route = warp::post()
        .and(warp::path("which_species_is_this"))
        .and(warp::body::json())
        .and(warp::addr::remote())
        .and_then(move |body: InputBody, addr: Option<std::net::SocketAddr>| {
            let log = log.clone();
            async move {
                let log_guard = log.lock().await; // Lock for atomic logging
                if let Some(ip) = addr {
                    info!("Incoming request from {}", ip);
                }
                info!("Request body: {:?}", body);

                let result = handle_sum(body).await;

                match &result {
                    Ok(response) => (),
                    Err(_) => warn!("Request processing failed"),
                }

                drop(log_guard); // Explicitly drop the lock

                result
            }
        });

    warp::serve(sum_route)
        .run(([0, 0, 0, 0], 8080))
        .await;
}

async fn handle_sum(body: InputBody) -> Result<impl warp::Reply, warp::Rejection> {
    if body.features.len() != 4 {
        return Err(warp::reject::custom(InvalidLengthError));
    }

    // Use the classify function to process the numbers.
    let classification_result = classify(body.features[0], body.features[1], body.features[2], body.features[3]);

    let result = ResultBody {
        result: classification_result,
    };
    info!("Classification result: {}", classification_result);
    Ok(warp::reply::json(&result))
}

#[derive(Deserialize, Debug)]
struct InputBody {
    features: Vec<f32>,
}

#[derive(Serialize, Debug)]
struct ResultBody {
    result: i32,
}

#[derive(Debug)]
struct InvalidLengthError;

impl warp::reject::Reject for InvalidLengthError {}

// Custom error response
impl warp::Reply for InvalidLengthError {
    fn into_response(self) -> warp::reply::Response {
        warp::reply::with_status("Invalid input length: exactly 4 numbers are required.", StatusCode::BAD_REQUEST).into_response()
    }
}
