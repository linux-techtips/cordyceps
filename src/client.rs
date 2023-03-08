//! # Client
//!

use bytes::Bytes;
pub use futures_util::stream::{Stream, StreamExt};
use reqwest::Client as ReqwestClient;
pub use reqwest::Result as ReqwestResult;
use serde::Serialize;

/// A Generic Error that will hopefully become more specific in the future.
pub type Error = Box<dyn std::error::Error + std::marker::Send + std::marker::Sync>;

/// A wrapper around [`Client`](Client) that is specific to chat [`Payloads`](crate::chat::Payload).
#[cfg(feature = "chat")]
pub struct ChatClient(Client<crate::chat::Payload>);

#[cfg(feature = "chat")]
impl ChatClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(Client::new(api_key.into(), crate::chat::API_URL))
    }

    pub async fn send(
        &self,
        payload: &crate::chat::Payload,
    ) -> Result<impl Stream<Item = ReqwestResult<Bytes>>, Error> {
        self.0.send(payload).await
    }
}

/// A generic client for sending json payloads to OpenAi's API.
pub struct Client<P: Serialize + ?Sized> {
    api_key: String,
    api_url: String,

    marker: std::marker::PhantomData<P>,
}

impl<P: Serialize + ?Sized> Client<P> {
    pub fn new(api_key: impl Into<String>, api_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_url: api_url.into(),
            marker: std::marker::PhantomData,
        }
    }

    /// Sends a payload to the API. Returns a stream of bytes that can be asynchronously awaited.
    pub async fn send(
        &self,
        payload: &P,
    ) -> Result<impl Stream<Item = ReqwestResult<Bytes>>, Error> {
        let req = ReqwestClient::new()
            .post(&self.api_url)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await?;

        if !req.status().is_success() {
            return Err(format!(
                "Could not request openai with status code: {}",
                req.status()
            )
            .into());
        }

        let resp = req.bytes_stream().filter_map(|result| async move {
            match result {
                Ok(bytes) => Some(Ok(bytes.slice(6..))), // Removes the b"data: " prefix. Thank you
                // openai!
                Err(_) => Some(result),
            }
        });

        Ok(Box::pin(resp))
    }
}
