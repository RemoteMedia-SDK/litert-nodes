//! LiteRT and LiteRT-LM for RemoteMedia nodes, as a library: the runtimes are
//! loaded when a model starts (`libloading`), not linked.
//!
//! * `litert`: LiteRT's TFLite interpreter C API ([`litert::TfLiteBackend`]);
//! * `compiled`: LiteRT's CompiledModel API, for GPU / NPU ([`compiled::CompiledBackend`]);
//! * `backend`: the [`backend::Backend`] both implement;
//! * `lm`: LiteRT-LM's C API: an engine for a `.litertlm` language model and
//!   conversations that stream replies;
//! * `container`: the `.litertlm` bundle format;
//! * `bilstm`: the bidirectional LSTM custom op some bundles need.
//!
//! No plugin export and no C++ build: another plugin or application links
//! this crate without exporting the LiteRT plugin's factories.

pub mod backend;
pub mod bilstm;
pub mod compiled;
pub mod container;
pub mod litert;
pub mod lm;
