use async_openai::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent,
};

use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::io::{AsyncWriteExt, stdout};

use async_openai::{Client, config::OpenAIConfig, types::CreateChatCompletionRequestArgs};

use crate::{Message::*, Query, SpnlResult};

#[cfg(feature = "rag")]
use crate::augment::embed::EmbedData;

pub enum Provider {
    OpenAI,
    Gemini,
    Ollama,
}

fn api_base(provider: &Provider) -> String {
    ::std::env::var("OPENAI_API_BASE").unwrap_or_else(|_| {
        {
            match provider {
                // Note: NO TRAILING SLASHES!
                Provider::OpenAI => "https://api.openai.com/v1",
                Provider::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
                Provider::Ollama => "http://localhost:11434/v1",
            }
        }
        .into()
    })
}

pub async fn generate(
    provider: Provider,
    model: &str,
    input: &Query,
    max_tokens: &Option<i32>,
    temp: &Option<f32>,
    m: Option<&MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    if prepare {
        todo!()
    }

    let client = Client::with_config(OpenAIConfig::new().with_api_base(api_base(&provider)));
    let input_messages = messagify(input);

    let quiet = m.is_some();
    let mut stdout = stdout();
    /* if !quiet {
        if let Some(ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(content),
            ..
        })) = input_messages.last()
        {
            stdout.write_all(b"\x1b[1mUser: \x1b[0m").await?;
            stdout.write_all(content.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
        }
    } */

    // Extract a max tokens
    let mt = max_tokens
        .map(|mt| match mt {
            0 => 2048_u32, // vllm 400's if given 0
            _ => mt as u32,
        })
        .unwrap_or(2048);

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(input_messages.clone())
        .temperature(temp.unwrap_or_default())
        .max_tokens(mt)
        .max_completion_tokens(mt)
        .build()?;

    // Debug: print the request payload as JSON
    if std::env::var("RUST_LOG").unwrap_or_default().contains("debug") || std::env::var("VERBOSE").unwrap_or_default() == "1" {
        match serde_json::to_string(&request) {
            Ok(json) => eprintln!("[DEBUG] Request payload: {}", json),
            Err(e) => eprintln!("[DEBUG] Failed to serialize request: {}", e),
        }
    }

    let style = ProgressStyle::with_template(
        "{msg} {wide_bar:.yellow/orange} {pos:>7}/{len:7} [{elapsed_precise}]",
    )?;
    let mut pb = m.map(|m| {
        m.add(
            max_tokens
                .map(|max_tokens| ProgressBar::new((max_tokens as u64) * 4))
                .unwrap_or_else(ProgressBar::no_length)
                .with_style(style)
                .with_message("Generating"),
        )
    });

    // println!("A {:?}", client.models().list().await?);

    let mut response_string = String::new();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
    }

    if std::env::var("RUST_LOG").unwrap_or_default().contains("debug") || std::env::var("VERBOSE").unwrap_or_default()=="1" {
        eprintln!("[DEBUG] Creating chat stream to {}", api_base(&provider));
    }
    let mut stream = client.chat().create_stream(request).await?;
    loop {
        match stream.next().await {
            Some(Ok(res)) => {
                for chat_choice in res.choices.iter() {
                    if let Some(ref content) = chat_choice.delta.content {
                        if !quiet {
                            stdout.write_all(b"\x1b[32m").await?; // green
                            stdout.write_all(content.as_bytes()).await?;
                            stdout.flush().await?;
                            stdout.write_all(b"\x1b[0m").await?; // reset color
                        } else if let Some(pb) = pb.as_mut() {
                            pb.inc(content.len() as u64)
                        }
                        response_string += content.as_str();
                    }
                }
            }
            Some(Err(err)) => return Err(err.into()),
            None => break,
        }
    }
    if !quiet {
        stdout.write_all(b"\n").await?;
    }

    Ok(Query::Message(Assistant(response_string)))
}

pub fn messagify(input: &Query) -> Vec<ChatCompletionRequestMessage> {
    match input {
        Query::Seq(v) | Query::Plus(v) | Query::Cross(v) => {
            let iter = v.iter().flat_map(messagify);
            iter.collect()
        },
        Query::Message(System(s)) => vec![ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                name: None,
                content: ChatCompletionRequestSystemMessageContent::Text(s.clone()),
            },
        )],
        Query::Message(Assistant(s)) => vec![ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                name: None,
                refusal: None,
                audio: None,
                tool_calls: None,
                #[allow(deprecated)]
                function_call: None,
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                    s.clone(),
                )),
            },
        )],
        o => {
            let s = o.to_string();
            if s.is_empty() {
                vec![]
            } else {
                vec![ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        name: None,
                        content: ChatCompletionRequestUserMessageContent::Text(o.to_string()),
                    },
                )]
            }
        }
    }
}

#[cfg(feature = "rag")]
pub fn contentify(input: &Query) -> Vec<String> {
    match input {
        Query::Seq(v) | Query::Plus(v) | Query::Cross(v) => {
            let iter = v.iter().flat_map(contentify);
            iter.collect()
        },
        Query::Message(Assistant(s)) | Query::Message(System(s)) => vec![s.clone()],
        o => {
            let s = o.to_string();
            if s.is_empty() {
                vec![]
            } else {
                vec![o.to_string()]
            }
        }
    }
}

#[cfg(feature = "rag")]
pub async fn embed(
    provider: Provider,
    embedding_model: &str,
    data: &EmbedData,
) -> anyhow::Result<impl Iterator<Item = Vec<f32>> + use<>> {
    use async_openai::types::CreateEmbeddingRequestArgs;

    let client = Client::with_config(OpenAIConfig::new().with_api_base(api_base(&provider)));

    let docs = match data {
        EmbedData::String(s) => &vec![s.clone()],
        EmbedData::Vec(v) => v,
        EmbedData::Query(u) => &contentify(u),
    };

    let request = CreateEmbeddingRequestArgs::default()
        .model(embedding_model)
        .input(docs)
        .build()?;

    if std::env::var("RUST_LOG").unwrap_or_default().contains("debug") || std::env::var("VERBOSE").unwrap_or_default()=="1" {
        eprintln!("[DEBUG] Creating embeddings request to {}", api_base(provider));
    }
    Ok(client
        .embeddings()
        .create(request)
        .await?
        .data
        .into_iter()
        .map(|e| e.embedding))
}
