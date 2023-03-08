//! # Chat
//!
//! The chat module contains the required data structures for [`OpenAi's Chat Completion`](https://platform.openai.com/docs/api-reference/completions/create)
//! This can also be used as documentation for OpenAi's streaming chat completions API that they
//! must have forgotten to document.
//!
//! ## Response usage
//!
//! The [`Response`](Response) is used to be Deserialized from JSON. Sometimes OpenAi doesn't like
//! to follow the JSON spec so I recommend using the [`serde_jsonrc`](https://docs.rs/serde_jsonrc)
//! crate for a more lenient JSON deserialization.
//!
use serde::{Deserialize, Serialize};

use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Result as FmtResult},
};

/// The API_URL is a public constant unique to each feature's module.
pub const API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Roles for chat completions.
/// - `Role::System`: Assignes a behavior to the assistant.
/// - `Role::User`: Instructs the assistant.
/// - `Role::Assistant`: Meant for storing previous responses.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

impl Serialize for Role {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(format!("{self}").as_str())
    }
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            _ => Err(serde::de::Error::custom(
                format!("{s} is not a valid role",),
            )),
        }
    }
}

/// The models available to use for chat completions.
/// - `Model::Gpt35Turbo`: OpenAI's most advanced model. Equivalent to [`ChatGPT`](https://chat.openai.com/chat).
/// - `Model::Gpt35Turbo0301`: Interchangable with `Model::Gpt3Turbo`.
#[derive(Debug, Clone, PartialEq)]
pub enum Model {
    Gpt35Turbo,
    Gpt35Turbo0301,
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Model::Gpt35Turbo => write!(f, "gpt-3.5-turbo"),
            Model::Gpt35Turbo0301 => write!(f, "gpt-3.5-turbo-0301"),
        }
    }
}

impl Serialize for Model {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(format!("{self}").as_str())
    }
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "gpt-3.5-turbo" => Ok(Model::Gpt35Turbo),
            "gpt-3.5-turbo-0301" => Ok(Model::Gpt35Turbo0301),
            _ => Err(serde::de::Error::custom(format!(
                "{s} is not a valid model",
            ))),
        }
    }
}

/// Messages are used to prompt the chosen model. Used to assign content to the `Role`.
/// - `role`: The role to assign the message to.
/// - `content`: The content to assign to the message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// The payload contains all of the data needed to complete a chat.
/// See [`OpenAi's Completion Documentation`](https://platform.openai.com/docs/api-reference/completions/create) for more information on each field's meaning
/// It's not recommended to construct this directly. See [`PayloadBuilder`](PayloadBuilder) for
/// intended usage.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Payload {
    pub model: Model,
    pub messages: Vec<Message>,
    pub temperature: f64,
    pub top_p: f64,
    pub n: isize,
    pub stream: bool,
    pub stop: Option<String>,
    pub max_tokens: isize,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub logit_bias: HashMap<String, f64>,
    pub user: String,
}

impl Payload {
    pub fn builder() -> PayloadBuilder {
        PayloadBuilder::default()
    }
}

/// Recommended contructor for payloads.
/// The only required field is `messages` which have several methods for setting.
#[derive(Debug, Clone)]
pub struct PayloadBuilder {
    model: Model,
    messages: Vec<Message>,
    temperature: f64,
    top_p: f64,
    n: isize,
    stream: bool,
    stop: Option<String>,
    max_tokens: isize,
    presence_penalty: f64,
    frequency_penalty: f64,
    logit_bias: HashMap<String, f64>,
    user: String,
}

impl Default for PayloadBuilder {
    fn default() -> Self {
        Self {
            model: Model::Gpt35Turbo,
            messages: vec![],
            temperature: 1.0,
            top_p: 1.0,
            n: 1,
            stream: true,
            stop: None,
            max_tokens: 1024,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: HashMap::new(),
            user: "Rust Openai Developer".to_string(),
        }
    }
}

impl PayloadBuilder {
    pub fn build(self) -> Result<Payload, Box<dyn std::error::Error + Sync + Send>> {
        if self.messages.is_empty() {
            return Err("messages are not set".into());
        }
        Ok(Payload {
            model: self.model,
            messages: self.messages,
            temperature: self.temperature,
            top_p: self.top_p,
            n: self.n,
            stream: self.stream,
            stop: self.stop,
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            logit_bias: self.logit_bias,
            user: self.user,
        })
    }

    pub fn model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        messages.into_iter().for_each(|m| self.messages.push(m));
        self
    }

    pub fn message(mut self, message: impl Into<Message>) -> Self {
        self.messages.push(message.into());
        self
    }

    pub fn user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::new(Role::User, content));
        self
    }

    pub fn system_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::new(Role::System, content));
        self
    }

    pub fn assistant_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::new(Role::Assistant, content));
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn n(mut self, n: isize) -> Self {
        self.n = n;
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn stop(mut self, stop: impl Into<String>) -> Self {
        let _ = self.stop.insert(stop.into());
        self
    }

    pub fn max_tokens(mut self, max_tokens: isize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn presence_penalty(mut self, presence_penalty: f64) -> Self {
        self.presence_penalty = presence_penalty;
        self
    }

    pub fn frequency_penalty(mut self, frequency_penalty: f64) -> Self {
        self.frequency_penalty = frequency_penalty;
        self
    }

    pub fn logit_bias(mut self, logit_bias: HashMap<String, f64>) -> Self {
        self.logit_bias = logit_bias;
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = user.into();
        self
    }
}

impl From<Payload> for serde_jsonrc::Value {
    fn from(val: Payload) -> Self {
        // SAFETY: Something has gone horribly wrong if we cannot serialize the payload.
        serde_jsonrc::to_string(&val).unwrap().into()
    }
}

/// When streaming, OpenAI will return the state of the stream in this enum.
/// These are effectively useless as the response from the client can be iterated over.
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "length" => Ok(FinishReason::Length),
            "stop" => Ok(FinishReason::Stop),
            _ => Err(serde::de::Error::custom(format!(
                "{s} is not a valid finish reason",
            ))),
        }
    }
}

/// I don't know what OpenAi was on when they designed this, but the response content is stored in
/// this struct. See [`Choice`](Choice) for why this is dumb.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Delta {
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Choice {
    pub delta: Delta,
    pub index: isize,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Response {
    pub id: String,
    pub object: String,
    pub created: isize,
    pub model: Model,
    pub choices: Vec<Choice>,
}

impl Response {
    pub fn text(self, n: usize) -> Option<String> {
        self.choices.into_iter().nth(n).map(|c| c.delta.content)
    }
}

#[cfg(test)]
mod tests {
    #[ignore]
    #[test]
    fn test_payload() {
        todo!()
    }

    #[ignore]
    #[test]
    fn test_payload_serialize() {
        todo!()
    }

    #[ignore]
    #[test]
    fn test_response() {
        todo!()
    }

    #[ignore]
    #[test]
    fn test_response_deserialize() {
        todo!()
    }
}
