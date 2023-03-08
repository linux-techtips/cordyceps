//! # openai
//!
//! The cordyceps crate is a **fast** and **reliable**  API for OpenAi's various AI models
//!
//! ## Supported Features
//!
//! - [`Chat Completion`](crate::chat)
//!
//! ## Planned Features
//! - [`Blocking API`](https://tenor.com/1lq0.gif)
//! - [`Image Generation`](https://platform.openai.com/docs/guides/images/introduction)
//! - [`Image Editing`](https://platform.openai.com/docs/guides/images/introduction)
//! - [`Image Variation`](https://platform.openai.com/docs/guides/images/introduction)
//! - [`Audio Transcription`](https://platform.openai.com/docs/guides/speech-to-text)
//! - [`Audio Translation`](https://platform.openai.com/docs/guides/speech-to-text)
//! - [`Text Moderation`](https://platform.openai.com/docs/guides/moderation)
//!
//! ## Example
//!
//! A small example of how to use the openai crate
//! ```
//! use cordyceps_api::client::{ChatClient, Error, StreamExt};
//! use cordyceps_api::chat::{Payload, Response};
//! use tokio::io::AsyncWriteExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Error> {
//!     let mut stdout = tokio::io::stdout();
//!     let api_key = std::env::var("OPENAI_API_KEY")?;
//!     let payload = Payload::builder()
//!         .system_message("Implement the Rust programming language into your responses")
//!         .user_message("Tell me a joke")
//!         .build()?;
//!
//!     let client = ChatClient::new(&api_key);
//!     let mut response = client.send(&payload).await?;
//!
//!     while let Some(chunk) = response.next().await {
//!         let body = chunk.unwrap();
//!         match serde_jsonrc::from_slice::<Response>(&body) {
//!             Ok(r) => {
//!                 let text = r.text(0).unwrap();
//!                 stdout.write_all(text.as_bytes()).await.unwrap();
//!                 stdout.flush().await.unwrap();
//!             },
//!             Err(_) => continue,
//!         };
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Re-exports
//! `pub use futures_util::stream::StreamExt;`
//! `pub use futures_util::stream::Stream;`
//! `pub use reqwest::Result as ReqwestResult;`

#[cfg(not(feature = "blocking"))]
pub mod client;

#[cfg(feature = "chat")]
pub mod chat;

#[cfg(feature = "tests")]
#[cfg(test)]
mod tests {
    use crate::chat::{Payload, Response};
    use crate::client::{ChatClient, StreamExt};
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn example() {
        let mut stdout = tokio::io::stdout();
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let payload = Payload::builder()
            .system_message("Implement the Rust programming language into your responses")
            .user_message("Tell me a joke")
            .build()
            .unwrap();

        let client = ChatClient::new(&api_key);
        let mut response = client.send(&payload).await.unwrap();

        while let Some(chunk) = response.next().await {
            let body = chunk.unwrap();
            match serde_jsonrc::from_slice::<Response>(&body) {
                Ok(r) => {
                    let text = r.text(0).unwrap();
                    stdout.write_all(text.as_bytes()).await.unwrap();
                    stdout.flush().await.unwrap();
                }
                Err(_) => continue,
            };
        }
    }
}
