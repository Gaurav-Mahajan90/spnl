use indicatif::MultiProgress;
use tokio::io::{AsyncWriteExt, stdout};

use crate::{Generate, Message::Assistant, Query, SpnlResult, to_string};

#[derive(serde::Deserialize)]
struct Message {
    // role: String,
    content: String,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: Message,
}

#[derive(serde::Deserialize)]
struct Response {
    choices: Vec<Choice>,
}

pub async fn generate(
    model: &str,
    input: &Query,
    max_tokens: &Option<i32>,
    temp: &Option<f32>,
    m: Option<&MultiProgress>,
    prepare: bool,
) -> SpnlResult {
    let exec = if prepare { "prepare" } else { "execute" };
    let client = reqwest::Client::new();

    let query = Query::Generate(Generate {
        model: model.to_string(),
        input: Box::new(input.clone()),
        max_tokens: *max_tokens,
        temperature: *temp,
    });
    // eprintln!("Sending query {:?}", to_string(&query)?);

    let url = format!("http://localhost:8000/v1/query/{exec}");
    let body = to_string(&query)?;
    if std::env::var("RUST_LOG").unwrap_or_default().contains("debug") || std::env::var("VERBOSE").unwrap_or_default()=="1" {
        eprintln!("[DEBUG] POST {}", url);
        eprintln!("[DEBUG] Body: {}", body);
    }
    let response = client
        .post(url)
        .header("Content-Type", "text/plain")
        .body(body)
        .send()
        .await?;

    let response_string = if prepare {
        "prepared".to_string()
    } else {
        response.json::<Response>().await?.choices[0]
            .message
            .content
            .clone()
    };

    let quiet = m.is_some();
    let mut stdout = stdout();
    if !quiet {
        stdout.write_all(b"\x1b[1mAssistant: \x1b[0m").await?;
        stdout.write_all(response_string.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
    }

    Ok(Query::Message(Assistant(response_string)))
}
