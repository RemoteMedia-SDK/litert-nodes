//! Real-model run: LiteRT `whisper-tiny` driving the shared streaming state
//! (`remotemedia_traits::stt::WhisperStreamingState`, which runs on
//! `remotemedia_modality::stt::TranscriptStabilizer`).
//!
//! Ignored by default. Run with:
//!
//! ```sh
//! cargo test --test stt_stream_real -- --ignored --nocapture
//! ```
//!
//! Env (defaults assume the sibling-checkout layout this repository already
//! uses for `../remotemedia-plugin-sdk`):
//! - `RM_WHISPER_TFLITE`: Whisper `.tflite`, with `tokenizer.json` beside it
//!   (default `../models/whisper-tiny-litert/whisper_tiny_30s_f32.tflite`)
//! - `RM_LITERT_CPU_LIB`: LiteRT / TFLite C library for the CPU interpreter
//!   (default: LiteRT's own library search)
//! - `RM_STT_CLIPS_DIR`: directory with the example clips
//!   (default `../remotemedia-sdk/examples`)
//!
//! LiteRT decodes with `<|notimestamps|>`, so each window comes back as one
//! segment spanning the whole window. The stabilizer spreads those times
//! evenly over the words and lines windows up by word overlap: the hard,
//! timestamp-poor case.

use std::sync::Mutex;

use remotemedia_litert_nodes::mel::to_mono_16k;
use remotemedia_litert_nodes::{Whisper as LiteWhisper, WhisperConfig as LiteConfig};
use remotemedia_modality::stt::{HypothesisScope, TranscriptStabilizer, TranscriptStreamingConfig};
use remotemedia_plugin_sdk::traits::stt::{
    DecodedSegment, TranscriptEvent, TranscriptStreamingMode, TranscriptUpdate, Whisper,
    WhisperConfig, WhisperStreamingState,
};
use remotemedia_plugin_sdk::types::Error;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// A path beside this checkout, e.g. `sibling("models")` → `../models`.
fn sibling(path: &str) -> String {
    format!("{}/../{path}", env!("CARGO_MANIFEST_DIR"))
}

fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / scale)
                .collect()
        }
    };
    to_mono_16k(&samples, spec.sample_rate, spec.channels as u32)
}

struct LiteRtWhisper {
    model: Mutex<LiteWhisper>,
    config: WhisperConfig,
}

// SAFETY: the `Whisper` trait requires `Send + Sync`, but the LiteRT
// interpreter is not `Send`. This test creates, uses and drops each model on
// the test thread only and never shares it, so asserting `Send`/`Sync` here
// cannot move the interpreter across threads. (`WhisperLiteRtNode` keeps its
// model on a dedicated thread for the same reason.)
unsafe impl Send for LiteRtWhisper {}
unsafe impl Sync for LiteRtWhisper {}

impl Whisper for LiteRtWhisper {
    fn config(&self) -> WhisperConfig {
        self.config.clone()
    }

    fn decode_window(&self, samples: &[f32]) -> Result<Vec<DecodedSegment>, Error> {
        let transcript = self
            .model
            .lock()
            .unwrap()
            .transcribe(samples)
            .map_err(Error::Execution)?;
        Ok(vec![DecodedSegment {
            text: transcript.text,
            start_ms: 0,
            end_ms: samples.len() as u64 * 1000 / 16_000,
        }])
    }
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn wer(reference: &str, hypothesis: &str) -> f64 {
    let (r, h) = (words(reference), words(hypothesis));
    let mut d: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut prev = d[0];
        d[0] = i;
        for j in 1..=h.len() {
            let cur = (d[j] + 1)
                .min(d[j - 1] + 1)
                .min(prev + usize::from(r[i - 1] != h[j - 1]));
            prev = d[j];
            d[j] = cur;
        }
    }
    d[h.len()] as f64 / r.len().max(1) as f64
}

/// Check the transcript-updates contract; return the final's text.
fn check_updates(updates: &[TranscriptUpdate], mode: TranscriptStreamingMode) -> String {
    let (mut shown, mut stable, mut last_rev) = (Vec::<String>::new(), 0usize, 0u64);
    let mut final_text = None;
    for update in updates {
        assert!(update.revision() > last_rev, "revision did not increase");
        last_rev = update.revision();
        match update {
            TranscriptUpdate::Delta(e) | TranscriptUpdate::Correction(e) => {
                assert!(
                    mode != TranscriptStreamingMode::Off,
                    "off mode emitted an edit"
                );
                if mode == TranscriptStreamingMode::Stable {
                    assert!(
                        matches!(update, TranscriptUpdate::Delta(_)),
                        "stable corrected"
                    );
                }
                assert_eq!(
                    e.index + e.removed,
                    shown.len(),
                    "edit does not reach the end"
                );
                assert!(e.index >= stable, "edit touched stable tokens");
                assert!(e.stable_tokens >= stable, "stable_tokens went down");
                stable = e.stable_tokens;
                shown.truncate(e.index);
                shown.extend(e.tokens.iter().map(|t| t.text.clone()));
                assert_eq!(shown.join(" "), e.transcript);
            }
            TranscriptUpdate::Final(f) => {
                if mode != TranscriptStreamingMode::Off {
                    let tokens: Vec<_> = f.tokens.iter().map(|t| t.text.clone()).collect();
                    assert_eq!(tokens, shown, "replay rule");
                }
                assert!(final_text.is_none(), "two finals");
                final_text = Some(f.text.clone());
            }
        }
    }
    final_text.expect("no final")
}

fn stream(whisper: &LiteRtWhisper, audio: &[f32]) -> (Vec<TranscriptEvent>, Vec<TranscriptUpdate>) {
    let mut state = WhisperStreamingState::default();
    let (mut events, mut updates) = (Vec::new(), Vec::new());
    for chunk in audio.chunks(8_000) {
        events.extend(whisper.stream_audio(&mut state, chunk, 16_000, 1).unwrap());
        updates.extend(state.take_updates());
    }
    events.extend(whisper.finalize(&mut state).unwrap());
    updates.extend(state.take_updates());
    (events, updates)
}

fn lite_config() -> LiteConfig {
    let model_path = env_or(
        "RM_WHISPER_TFLITE",
        &sibling("models/whisper-tiny-litert/whisper_tiny_30s_f32.tflite"),
    );
    let mut lite = LiteConfig::new(model_path);
    lite.accelerator = "cpu".into();
    lite.threads = 8;
    lite.runtime_lib = std::env::var("RM_LITERT_CPU_LIB").ok();
    lite
}

/// (clip, sliding-window bound in ms). Both clips are longer than the window.
const CLIPS: [(&str, u32); 2] = [
    ("question-with-pauses.wav", 6_000),
    ("audio_examples/VOXTAB_Medical_audio_trim.wav", 10_000),
];

fn load_clip(clip: &str) -> Vec<f32> {
    let clips = env_or("RM_STT_CLIPS_DIR", &sibling("remotemedia-sdk/examples"));
    let mut audio = read_wav(&format!("{clips}/{clip}"));
    audio.truncate(29 * 16_000); // the model takes at most 30 s
    audio
}

const MODES: [TranscriptStreamingMode; 3] = [
    TranscriptStreamingMode::Full,
    TranscriptStreamingMode::Stable,
    TranscriptStreamingMode::Off,
];

/// The recommended path for a model without timestamps: re-decode the whole
/// utterance so far every 2 s and push it with `Utterance` scope. No audio is
/// ever trimmed, so nothing can be lost; the stabilizer only decides what to
/// show and commit.
#[test]
#[ignore = "needs the LiteRT whisper-tiny model and runtime; run with --ignored"]
fn litert_whisper_utterance_redecode() {
    let lite = lite_config();
    let mut model = LiteWhisper::load(&lite).expect("load whisper-tiny");
    println!("runtime: {}", model.runtime);
    for (clip, _) in CLIPS {
        let audio = load_clip(clip);
        let one_shot = model.transcribe(&audio).unwrap().text;
        println!(
            "
== {clip} ({:.1} s), utterance re-decode every 2 s",
            audio.len() as f64 / 16_000.0
        );
        println!("   one-shot: {one_shot}");
        for mode in MODES {
            let mut stabilizer = TranscriptStabilizer::new(TranscriptStreamingConfig {
                mode,
                ..TranscriptStreamingConfig::default()
            });
            let started = std::time::Instant::now();
            let mut updates = Vec::new();
            for end in (32_000..audio.len()).step_by(32_000) {
                let text = model.transcribe(&audio[..end]).unwrap().text;
                updates.extend(stabilizer.push_text(&text, HypothesisScope::Utterance));
            }
            updates.extend(stabilizer.finalize_text(&one_shot, HypothesisScope::Utterance));
            let final_text = check_updates(&updates, mode);
            let score = wer(&one_shot, &final_text);
            println!(
                "   {mode:?}: delta {}, correction {} | {:.1}s | WER vs one-shot {score:.2}
     final: {final_text}",
                updates
                    .iter()
                    .filter(|u| matches!(u, TranscriptUpdate::Delta(_)))
                    .count(),
                updates
                    .iter()
                    .filter(|u| matches!(u, TranscriptUpdate::Correction(_)))
                    .count(),
                started.elapsed().as_secs_f64(),
            );
            assert!(
                score < 0.2,
                "streamed final drifted from the one-shot decode"
            );
        }
    }
}

/// The `WhisperStreamingState` sliding-window path with LiteRT's
/// timestamp-less decode (one segment spanning each window). Checks the
/// contract and that the legacy and shared finals agree. Accuracy is only
/// reported: without real segment timings the window cannot know where its
/// committed text ends in the audio, so trimming and the window-bound
/// force-commit can lose or commit the wrong words. Turn on timestamp tokens
/// before using this path in production.
#[test]
#[ignore = "needs the LiteRT whisper-tiny model and runtime; run with --ignored"]
fn litert_whisper_sliding_window_without_timestamps() {
    let lite = lite_config();
    let mut one_shot_model = LiteWhisper::load(&lite).expect("load whisper-tiny");
    for (clip, max_window_ms) in CLIPS {
        let audio = load_clip(clip);
        let one_shot = one_shot_model.transcribe(&audio).unwrap().text;
        println!(
            "
== {clip} ({:.1} s), sliding window {max_window_ms} ms",
            audio.len() as f64 / 16_000.0
        );
        println!("   one-shot: {one_shot}");
        for mode in MODES {
            let config = WhisperConfig {
                language: "en".into(),
                streaming_decode_interval_ms: 2_000,
                streaming_max_window_ms: max_window_ms,
                streaming_overlap_ms: 1_000,
                transcript_updates: mode,
                ..WhisperConfig::default()
            };
            config.validate().unwrap();
            let whisper = LiteRtWhisper {
                model: Mutex::new(LiteWhisper::load(&lite).expect("load whisper-tiny")),
                config,
            };
            let started = std::time::Instant::now();
            let (events, updates) = stream(&whisper, &audio);
            let legacy_final = events.last().expect("no legacy events");
            assert_eq!(legacy_final.kind, "final");
            let shared_final = check_updates(&updates, mode);
            assert_eq!(
                shared_final, legacy_final.text,
                "legacy and shared finals differ"
            );
            let count = |kind: &str| events.iter().filter(|e| e.kind == kind).count();
            println!(
                "   {mode:?}: legacy {{partial {}, correction {}}} | {:.1}s | WER vs one-shot {:.2} (reported only)
     final: {shared_final}",
                count("partial"),
                count("correction"),
                started.elapsed().as_secs_f64(),
                wer(&one_shot, &shared_final),
            );
        }
    }
}
