//! Voice runtime protocol for the Genie ecosystem.
//!
//! This crate defines the boundary between the voice pipeline and the agent.
//! `genie-voice-runtime` owns audio capture, wake/VAD, STT, TTS, playback, and
//! streaming session events. `genie-claw` owns prompt policy, memory, tools,
//! smart-home intent, safety gates, audit, and conversation state.

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeInfo {
    pub protocol_version: u32,
    pub runtime_name: String,
    pub runtime_version: String,
    pub capabilities: VoiceCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoiceCapabilities {
    pub wake_word: bool,
    pub push_to_talk: bool,
    pub streaming_stt: bool,
    pub streaming_tts: bool,
    pub speaker_identity: bool,
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceSessionConfig {
    pub session_id: String,
    pub capture_device: String,
    pub playback_device: String,
    pub sample_rate_hz: u32,
    pub stt_language: String,
    pub record_secs: u32,
    pub continuous: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transcript {
    pub session_id: String,
    pub text: String,
    pub language: Option<String>,
    pub speaker_id: Option<String>,
    pub final_segment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeakRequest {
    pub session_id: String,
    pub text: String,
    pub language: Option<String>,
    pub voice_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum VoiceEvent {
    RuntimeReady(RuntimeInfo),
    SessionStarted { session_id: String },
    WakeDetected { session_id: String, score: f32 },
    Transcript(Transcript),
    SpeakStarted { session_id: String },
    SpeakFinished { session_id: String },
    SessionFinished { session_id: String },
    Error(VoiceRuntimeError),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoiceRuntimeError {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum VoiceCommand {
    StartSession(VoiceSessionConfig),
    StopSession { session_id: String },
    Speak(SpeakRequest),
    Shutdown,
}

impl RuntimeInfo {
    pub fn new(runtime_name: impl Into<String>, runtime_version: impl Into<String>) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            runtime_name: runtime_name.into(),
            runtime_version: runtime_version.into(),
            capabilities: VoiceCapabilities::default(),
        }
    }
}
impl Default for VoiceCapabilities {
    fn default() -> Self {
        Self {
            wake_word: false,
            push_to_talk: true,
            streaming_stt: false,
            streaming_tts: false,
            speaker_identity: false,
            languages: vec!["auto".into()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_round_trips_as_json() {
        let transcript = Transcript {
            session_id: "voice-1".into(),
            text: "turn on the kitchen light".into(),
            language: Some("en".into()),
            speaker_id: None,
            final_segment: true,
        };

        let json = serde_json::to_string(&transcript).unwrap();
        let parsed: Transcript = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed, transcript);
    }

    #[test]
    fn command_uses_stable_tagged_shape() {
        let command = VoiceCommand::StopSession {
            session_id: "voice-1".into(),
        };

        let json = serde_json::to_value(command).unwrap();

        assert_eq!(json["command"], "stop_session");
        assert_eq!(json["session_id"], "voice-1");
    }
}
