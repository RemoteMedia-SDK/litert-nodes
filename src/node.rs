//! `WhisperLiteRtNode`: audio in, transcript (Text) out; any other input is
//! ignored, so it can sit behind a node that also emits events.
//!
//! Params (JSON):
//!
//! | Key | Default | |
//! |-----|---------|---|
//! | `model_path` | required | Whisper `.tflite` with `encode` / `decode` signatures |
//! | `tokenizer_path` | `tokenizer.json` beside the model | Hugging Face tokenizer |
//! | `runtime_lib` | platform default | LiteRT library (also `REMOTEMEDIA_LITERT_LIB`) |
//! | `accelerator` | `gpu` | `gpu`, `npu`, `npu,gpu` (CompiledModel) or `cpu` (interpreter) |
//! | `accelerator_dir` | none | where the NPU dispatch / compiler plugin libraries are |
//! | `threads` | 4 | XNNPACK threads (`cpu`) |
//! | `language` | `en` | |
//! | `mask` | `additive` | decoder mask encoding (`additive` or `binary`) |
//! | `transcript_updates` | `{"mode": "off"}` | `stable` / `full`: emit the shared `delta` / `final` Json events (`remotemedia_modality::stt`) instead of Text |
//!
//! A model path may name an environment variable as `${NAME}`, so one
//! manifest can point at a different location on each runner.

use std::sync::mpsc;
use std::time::Instant;

use async_trait::async_trait;
use remotemedia_modality::stt::{TranscriptStabilizer, TranscriptStreamingConfig};
use remotemedia_plugin_sdk::{
    abi_stable::{
        sabi_trait::TD_Opaque,
        std_types::{RErr, ROk, RResult, RString},
    },
    adapter::StreamingNodeFfiAdapter,
    traits::AsyncStreamingNode,
    types::{Error, RuntimeData},
    FfiNodeBox, FfiNodeFactory, FfiNode_TO,
};
use serde_json::Value;

use remotemedia_plugin_sdk::params::expand_env as expand;

use crate::mel::to_mono_16k;
use crate::whisper::{MaskMode, Transcript, Whisper, WhisperConfig};

pub const NODE_TYPE: &str = "WhisperLiteRtNode";

struct Job {
    audio: Vec<f32>,
    reply: futures_channel::oneshot::Sender<Result<Transcript, String>>,
}

/// The model lives on its own thread (the interpreter is neither `Send` nor
/// reentrant); turns queue for it.
pub struct WhisperLiteRtNode {
    jobs: std::sync::Mutex<mpsc::Sender<Job>>,
    updates: TranscriptStreamingConfig,
    stabilizer: std::sync::Mutex<TranscriptStabilizer>,
}

/// `params.transcript_updates`, defaulting to `off`.
pub fn transcript_updates_from_params(params: &Value) -> Result<TranscriptStreamingConfig, String> {
    let config: TranscriptStreamingConfig = match params.get("transcript_updates") {
        None | Some(Value::Null) => TranscriptStreamingConfig::default(),
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| format!("invalid transcript_updates: {e}"))?,
    };
    config.validate()?;
    Ok(config)
}

pub fn config_from_params(params: &Value) -> Result<WhisperConfig, String> {
    let model_path = params["model_path"]
        .as_str()
        .map(expand)
        .filter(|p| !p.is_empty())
        .ok_or("WhisperLiteRtNode needs params.model_path")?;
    let mut config = WhisperConfig::new(model_path);
    if let Some(p) = params["tokenizer_path"].as_str() {
        config.tokenizer_path = expand(p);
    }
    if let Some(p) = params["runtime_lib"]
        .as_str()
        .map(expand)
        .filter(|p| !p.is_empty())
    {
        config.runtime_lib = Some(p);
    }
    if let Some(a) = params["accelerator"]
        .as_str()
        .map(expand)
        .filter(|a| !a.is_empty())
    {
        config.accelerator = a;
    }
    if let Some(d) = params["accelerator_dir"]
        .as_str()
        .map(expand)
        .filter(|d| !d.is_empty())
    {
        config.accelerator_dir = Some(d);
    }
    if let Some(t) = params["threads"].as_i64() {
        config.threads = t as i32;
    }
    if let Some(l) = params["language"].as_str() {
        config.language = l.to_string();
    }
    config.mask = match params["mask"].as_str() {
        None | Some("additive") => MaskMode::Additive,
        Some("binary") => MaskMode::Binary,
        Some(other) => return Err(format!("unknown mask '{other}'")),
    };
    Ok(config)
}

impl WhisperLiteRtNode {
    pub fn start(config: WhisperConfig) -> Result<Self, String> {
        Self::start_with_updates(config, TranscriptStreamingConfig::default())
    }

    pub fn start_with_updates(
        config: WhisperConfig,
        updates: TranscriptStreamingConfig,
    ) -> Result<Self, String> {
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<String, String>>();
        std::thread::Builder::new()
            .name("whisper-litert".into())
            .spawn(move || {
                let mut whisper = match Whisper::load(&config) {
                    Ok(w) => {
                        let _ = ready_tx.send(Ok(w.runtime.clone()));
                        w
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = jobs_rx.recv() {
                    let _ = job.reply.send(whisper.transcribe(&job.audio));
                }
            })
            .map_err(|e| e.to_string())?;
        let runtime = ready_rx
            .recv()
            .map_err(|_| "the model thread exited while loading".to_string())??;
        eprintln!("[{NODE_TYPE}] loaded with {runtime}");
        Ok(Self {
            jobs: std::sync::Mutex::new(jobs_tx),
            stabilizer: std::sync::Mutex::new(TranscriptStabilizer::new(updates.clone())),
            updates,
        })
    }

    /// Turn one utterance's transcript into what the node emits: one Text in
    /// `off` mode, or the shared `delta` / `final` Json events otherwise.
    fn render_utterance(&self, text: &str) -> Vec<RuntimeData> {
        if self.updates.mode.is_off() {
            return vec![RuntimeData::Text(text.to_string())];
        }
        let updates = self
            .stabilizer
            .lock()
            .unwrap()
            .finalize_text(text, remotemedia_modality::stt::HypothesisScope::Utterance);
        updates
            .iter()
            .filter_map(|update| self.updates.render(update))
            .collect()
    }

    async fn transcribe(&self, audio: Vec<f32>) -> Result<Transcript, Error> {
        let (reply, answer) = futures_channel::oneshot::channel();
        self.jobs
            .lock()
            .unwrap()
            .send(Job { audio, reply })
            .map_err(|_| Error::Execution("whisper model thread is gone".into()))?;
        answer
            .await
            .map_err(|_| Error::Execution("whisper model thread is gone".into()))?
            .map_err(Error::Execution)
    }
}

#[async_trait]
impl AsyncStreamingNode for WhisperLiteRtNode {
    fn node_type(&self) -> &str {
        NODE_TYPE
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        match data {
            RuntimeData::Audio {
                samples,
                sample_rate,
                channels,
                ..
            } => {
                let audio = to_mono_16k(&samples, sample_rate, channels);
                Ok(RuntimeData::Text(self.transcribe(audio).await?.text))
            }
            other => Err(Error::InvalidData(format!(
                "{NODE_TYPE} takes audio, not {}",
                other.data_type()
            ))),
        }
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        _session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let RuntimeData::Audio {
            samples,
            sample_rate,
            channels,
            ..
        } = data
        else {
            return Ok(0);
        };
        let audio = to_mono_16k(&samples, sample_rate, channels);
        let seconds = audio.len() as f64 / 16_000.0;
        let start = Instant::now();
        let transcript = self.transcribe(audio).await?;
        eprintln!(
            "[{NODE_TYPE}] {seconds:.2} s of audio in {:.0} ms (mel {:.0}, encode {:.0}, decode {:.0}, {} tokens): {:?}",
            start.elapsed().as_secs_f64() * 1000.0,
            transcript.mel_ms,
            transcript.encode_ms,
            transcript.decode_ms,
            transcript.tokens,
            transcript.text
        );
        if transcript.text.is_empty() {
            return Ok(0);
        }
        let outputs = self.render_utterance(&transcript.text);
        let count = outputs.len();
        for output in outputs {
            callback(output)?;
        }
        Ok(count)
    }
}

#[derive(Default)]
pub struct WhisperLiteRtFactory;

impl FfiNodeFactory for WhisperLiteRtFactory {
    fn node_type(&self) -> RString {
        NODE_TYPE.into()
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let params: Value = serde_json::from_str(params.as_str()).unwrap_or(Value::Null);
        let started = config_from_params(&params).and_then(|config| {
            WhisperLiteRtNode::start_with_updates(config, transcript_updates_from_params(&params)?)
        });
        match started {
            Ok(node) => ROk(FfiNode_TO::from_value(
                StreamingNodeFfiAdapter::new(node),
                TD_Opaque,
            )),
            Err(e) => RErr(format!("{NODE_TYPE}: {e}").into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(updates: TranscriptStreamingConfig) -> WhisperLiteRtNode {
        WhisperLiteRtNode {
            jobs: std::sync::Mutex::new(mpsc::channel().0),
            stabilizer: std::sync::Mutex::new(TranscriptStabilizer::new(updates.clone())),
            updates,
        }
    }

    #[test]
    fn off_mode_emits_one_text() {
        let config = transcript_updates_from_params(&json!({})).unwrap();
        let out = node(config).render_utterance("hello there");
        assert!(matches!(out.as_slice(), [RuntimeData::Text(t)] if t == "hello there"));
    }

    #[test]
    fn stable_mode_emits_delta_and_final_per_utterance() {
        let config =
            transcript_updates_from_params(&json!({"transcript_updates": {"mode": "stable"}}))
                .unwrap();
        let node = node(config);
        let first = node.render_utterance("hello there");
        let second = node.render_utterance("second one");
        let kinds: Vec<_> = first
            .iter()
            .map(|d| match d {
                RuntimeData::Json(v) => v["type"].as_str().unwrap().to_string(),
                other => panic!("expected Json, got {}", other.data_type()),
            })
            .collect();
        assert_eq!(kinds, ["delta", "final"]);
        let RuntimeData::Json(last) = second.last().unwrap() else {
            panic!()
        };
        assert_eq!(last["utterance_id"], 1);
        assert_eq!(last["text"], "second one");
    }

    #[test]
    fn bad_transcript_updates_is_rejected() {
        assert!(
            transcript_updates_from_params(&json!({"transcript_updates": {"mode": "loud"}}))
                .is_err()
        );
        assert!(transcript_updates_from_params(
            &json!({"transcript_updates": {"agreement_updates": 0}})
        )
        .is_err());
    }
}
